"""LoRA SFT of Muse Glimmer 30B on Fireworks Serverless Training.

No dataset is uploaded: rows are rendered locally and sent batch by batch through
the Training SDK (dataset uploads are disabled after the October 2026 incident).
Docs: https://docs.fireworks.ai/fine-tuning/training-api/serverless

Setup (Python >= 3.11):
    uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python fireworks-training-cookbook
    export FIREWORKS_API_KEY=...     # in your shell only

Commands (run from the repo root):
    .venv/bin/python finetune/serverless_sft.py check      # local only: render, count tokens, estimate cost
    .venv/bin/python finetune/serverless_sft.py train      # PAID: trains, saves a checkpoint per epoch
    .venv/bin/python finetune/serverless_sft.py list --session <session-name>
    .venv/bin/python finetune/serverless_sft.py promote --session <session-name> --checkpoint e2-s23 --output-model <id>
    .venv/bin/python finetune/serverless_sft.py delete-session --session <session-name>

The training files are read as-is (never modified). Run metadata (ids, losses;
no training text) is written to finetune/runs/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"
REASONING_STRENGTH = "low"  # must match reasoning_effort sent at inference time
SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
TRAIN_FILE = Path("data/finetune/training/sft_train.jsonl")
VAL_FILE = Path("data/finetune/training/sft_val.jsonl")
RUNS_DIR = Path("finetune/runs")
TRAIN_PRICE_PER_M = 5.86  # USD per 1M train tokens, Muse Glimmer 30B (serverless pricing table)

# Hyperparameters chosen for this run.
EPOCHS = 2
LEARNING_RATE = 1e-4
LORA_RANK = 8
BATCH_SIZE = 32
MAX_SEQ_LEN = 2048  # longest rendered example is ~600 tokens; anything longer aborts rather than truncates
SEED = 0
CHECKPOINTS_PER_EPOCH = 5  # each one is saved (free) and scored on the val set (~20K tokens)


def _api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        sys.exit("FIREWORKS_API_KEY is not set in this shell.")
    return key


def load_rows(path: Path) -> list[list[dict]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def build_datums(rows, renderer):
    """Render chat rows into shifted next-token cross-entropy datums (loss on the assistant reply)."""
    from fireworks.training import sdk
    from training.renderer import TrainOnWhat

    datums, total_tokens, loss_tokens = [], 0, 0
    for i, messages in enumerate(rows):
        model_input, weights = renderer.build_supervised_example(
            messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        tokens = [int(t) for t in model_input.to_ints()]
        w = [float(x) for x in weights.tolist()]
        if len(tokens) > MAX_SEQ_LEN:
            sys.exit(f"row {i} renders to {len(tokens)} tokens > MAX_SEQ_LEN {MAX_SEQ_LEN}; refusing to truncate")
        if not any(x > 0 for x in w[1:]):
            sys.exit(f"row {i} has no assistant tokens to train on")
        datums.append(sdk.Datum(
            model_input=sdk.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"target_tokens": tokens[1:], "weights": w[1:]},
        ))
        total_tokens += len(tokens) - 1
        loss_tokens += sum(1 for x in w[1:] if x > 0)
    return datums, total_tokens, loss_tokens


def load_data():
    from training.renderer.muse_glimmer import MuseGlimmerOptions, MuseGlimmerRenderer
    from training.renderer.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    # The chat template appends "Reasoning strength: <x>." to the system prompt (default "high").
    # Train with the same strength the model is called with: reasoning_effort=REASONING_STRENGTH.
    renderer = MuseGlimmerRenderer(tokenizer, options=MuseGlimmerOptions(reasoning_strength=REASONING_STRENGTH))
    train = build_datums(load_rows(TRAIN_FILE), renderer)
    val = build_datums(load_rows(VAL_FILE), renderer)
    return train, val


def _weights(datum) -> list[float]:
    ws = datum.loss_fn_inputs["weights"]  # stored as TensorData by the SDK
    return [float(x) for x in (ws.tolist() if hasattr(ws, "tolist") else ws)]


def val_loss(training_client, datums) -> float:
    """Token-weighted mean NLL on held-out datums via forward only (no weight update)."""
    nll, n = 0.0, 0.0
    for start in range(0, len(datums), BATCH_SIZE):
        batch = datums[start:start + BATCH_SIZE]
        out = training_client.forward(batch, "cross_entropy").result()
        for i, d in enumerate(batch):
            lps = out.loss_fn_outputs[i]["logprobs"].to_torch().tolist()
            ws = _weights(d)
            nll -= sum(lp * w for lp, w in zip(lps, ws))
            n += sum(ws)
    return nll / max(n, 1.0)


def cmd_check(_args) -> None:
    (train, t_tok, t_loss), (val, v_tok, v_loss) = load_data()
    cost = t_tok * EPOCHS / 1e6 * TRAIN_PRICE_PER_M
    print(f"train: {len(train)} rows, {t_tok:,} tokens ({t_loss:,} assistant tokens)")
    print(f"val:   {len(val)} rows, {v_tok:,} tokens ({v_loss:,} assistant tokens)")
    print(f"plan:  {EPOCHS} epochs, lr {LEARNING_RATE}, rank {LORA_RANK}, batch {BATCH_SIZE} "
          f"-> {EPOCHS * -(-len(train) // BATCH_SIZE)} optimizer steps")
    steps = -(-len(train) // BATCH_SIZE)
    n_val = EPOCHS * len(checkpoint_steps(steps)) + 1
    print(f"checkpoints: steps {sorted(checkpoint_steps(steps))} of each epoch -> {n_val - 1} total")
    print(f"estimated training cost: ${cost:.2f} + {n_val} val passes x {v_tok:,} tokens "
          f"(${n_val * v_tok / 1e6 * 1.96:.2f}-${n_val * v_tok / 1e6 * TRAIN_PRICE_PER_M:.2f})")


def cmd_train(_args) -> None:
    from fireworks.training.sdk import AdamParams, FiretitanServiceClient

    (train, t_tok, _), (val, _, _) = load_data()
    print(f"rendered {len(train)} train / {len(val)} val rows; ~${t_tok * EPOCHS / 1e6 * TRAIN_PRICE_PER_M:.2f} to train")

    service = FiretitanServiceClient(api_key=_api_key(), base_url=SERVERLESS_URL)
    client = service.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNS_DIR / f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    log = {
        "session_name": service.training_session_name,
        "session_id": service.training_session_id,
        "run_id": client.run_id,
        "base_model": BASE_MODEL,
        "hyperparameters": {"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "lora_rank": LORA_RANK,
                            "batch_size": BATCH_SIZE, "seed": SEED},
        "steps": [], "checkpoints": [],
    }

    def save_log() -> None:
        log_path.write_text(json.dumps(log, indent=2))

    print(f"session={service.training_session_name} run={client.run_id}\nlog: {log_path}")
    save_log()

    log["base_val_loss"] = val_loss(client, val)
    print(f"base model val loss: {log['base_val_loss']:.4f}")
    save_log()

    adam = AdamParams(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, eps=1e-8,
                      weight_decay=0.0, grad_clip_norm=1.0)
    rng = random.Random(SEED)
    step = 0
    steps_per_epoch = -(-len(train) // BATCH_SIZE)
    save_at = checkpoint_steps(steps_per_epoch)
    print(f"checkpoints at steps {sorted(save_at)} of each {steps_per_epoch}-step epoch")
    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        for epoch_step, start in enumerate(range(0, len(order), BATCH_SIZE), start=1):
            batch = [train[i] for i in order[start:start + BATCH_SIZE]]
            t0 = time.time()
            fb = client.forward_backward(batch, "cross_entropy").result()
            client.optim_step(adam).result()
            step += 1
            metrics = getattr(fb, "metrics", None) or {}
            n_tok = sum(sum(1 for w in _weights(d) if w > 0) for d in batch)
            loss = metrics.get("loss:sum")
            mean = None if loss is None else float(loss) / max(n_tok, 1)
            log["steps"].append({"step": step, "epoch": epoch, "train_loss": mean})
            print(f"epoch {epoch}/{EPOCHS} step {step:03d} loss={'n/a' if mean is None else f'{mean:.4f}'} "
                  f"({time.time() - t0:.1f}s)")
            if epoch_step in save_at:
                name = f"e{epoch}-s{epoch_step:02d}"  # checkpoint names must be <= 17 chars
                sampler_path = client.save_weights_for_sampler(name).result().path
                v = val_loss(client, val)
                log["checkpoints"].append({"checkpoint": name, "epoch": epoch, "epoch_step": epoch_step,
                                           "global_step": step, "val_loss": v, "sampler_path": sampler_path})
                save_log()
                print(f"== {name} (global step {step}) val loss {v:.4f}")
        client.save_state(f"state-{epoch}").result()  # resumable state (weights + optimizer), once per epoch

    best = min(log["checkpoints"], key=lambda c: c["val_loss"])
    log["best_checkpoint"] = best["checkpoint"]
    save_log()
    print(f"\nDone. Lowest val loss: {best['checkpoint']} ({best['val_loss']:.4f}); base was {log['base_val_loss']:.4f}")
    print(f"Session left open so you can promote: {service.training_session_name}")
    print("Next: list -> promote the chosen checkpoint -> delete-session (frees the training slot).")


def checkpoint_steps(steps_per_epoch: int, per_epoch: int = CHECKPOINTS_PER_EPOCH) -> set[int]:
    """Evenly spaced steps within an epoch, always including the last step."""
    return {max(1, round(steps_per_epoch * k / per_epoch)) for k in range(1, per_epoch + 1)}


def _fw_client():
    from fireworks.training.sdk import FireworksClient
    return FireworksClient(api_key=_api_key())


def cmd_list(args) -> None:
    for row in _fw_client().list_training_session_checkpoints(args.session):
        print(f"{row.get('createTime')}  promotable={row.get('promotable')}  {row.get('name')}")


def cmd_promote(args) -> None:
    fw = _fw_client()
    rows = [r for r in fw.list_training_session_checkpoints(args.session)
            if r.get("promotable") and f"-{args.checkpoint}-" in r.get("checkpointName", "")]
    if len(rows) != 1:
        sys.exit(f"expected exactly one promotable checkpoint matching {args.checkpoint!r}, found {len(rows)}")
    model = fw.promote_session_checkpoint(name=rows[0]["name"], output_model_id=args.output_model,
                                          base_model=BASE_MODEL)
    print("promoted:", model)


def cmd_delete_session(args) -> None:
    import requests
    r = requests.delete(f"https://api.fireworks.ai/v1/{args.session}",
                        headers={"Authorization": f"Bearer {_api_key()}"}, timeout=60)
    print(r.status_code, r.text[:300])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("train")
    for name in ("list", "promote", "delete-session"):
        sp = sub.add_parser(name)
        sp.add_argument("--session", required=True, help="accounts/<acct>/trainingSessions/<id>")
        if name == "promote":
            sp.add_argument("--checkpoint", required=True, help="e.g. e2-s23 (see finetune/runs/*.json)")
            sp.add_argument("--output-model", required=True, help="1-63 chars: a-z, 0-9, hyphens")
    args = p.parse_args()
    {"check": cmd_check, "train": cmd_train, "list": cmd_list,
     "promote": cmd_promote, "delete-session": cmd_delete_session}[args.cmd](args)


if __name__ == "__main__":
    main()
