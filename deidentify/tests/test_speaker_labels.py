import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from de_identify_steplab import process_with_speaker_labels, split_speaker_label  # noqa: E402


def make_stats():
    return {
        "chunk_failures": 0, "line_failures": 0, "word_failures": 0,
        "redacted_span_expansions": 0, "redacted_extra_tokens": 0,
    }


def make_fake_api(names, seen):
    """Stands in for the model: redacts the given names and records every chunk it receives."""
    def api(post_text, prompt):
        seen.append(post_text)
        out = post_text
        for name in names:
            out = re.sub(r"\b%s\b" % re.escape(name), "[REDACTED]", out)
        return out
    return api


# --- splitting a line into label and body ---------------------------------------

def test_split_speaker_letter_label():
    assert split_speaker_label("Speaker A: Hi there.") == ("Speaker A:", "Hi there.")


def test_split_speaker_number_label():
    assert split_speaker_label("Speaker 2:Morning!") == ("Speaker 2:", "Morning!")


def test_split_role_labels():
    assert split_speaker_label("Coach: How did it go?") == ("Coach:", "How did it go?")
    assert split_speaker_label("Head Coach: Let's start.") == ("Head Coach:", "Let's start.")
    assert split_speaker_label("Teacher 2: Yes.") == ("Teacher 2:", "Yes.")


def test_line_without_label_is_returned_as_body():
    assert split_speaker_label("Just some text.") == (None, "Just some text.")


def test_speaker_with_a_name_is_not_a_label():
    # Only a single letter or number identifies a speaker; "Speaker Josh" is a name.
    assert split_speaker_label("Speaker Josh: hi") == (None, "Speaker Josh: hi")


def test_personal_name_is_not_a_label():
    assert split_speaker_label("Jordan: Hello there.") == (None, "Jordan: Hello there.")


# --- full pipeline: labels removed, body redacted, labels put back ---------------

def test_labels_are_restored_and_body_is_redacted():
    seen = []
    text = "Speaker A: Josh is here today.\nSpeaker B: Hi, Josh."
    result = process_with_speaker_labels(text, "prompt", make_fake_api(["Josh"], seen), make_stats())
    assert result == "Speaker A: [REDACTED] is here today.\nSpeaker B: Hi, [REDACTED]."


def test_labels_are_never_sent_to_the_model():
    seen = []
    text = "Speaker A: Josh is here.\nCoach: Okay."
    process_with_speaker_labels(text, "prompt", make_fake_api(["Josh"], seen), make_stats())
    sent = " ".join(seen)
    assert "Speaker" not in sent
    assert "Coach" not in sent


def test_label_is_kept_even_if_model_would_redact_it():
    seen = []
    text = "Speaker A: Okay then."
    result = process_with_speaker_labels(
        text, "prompt", make_fake_api(["Speaker", "A"], seen), make_stats()
    )
    assert result == "Speaker A: Okay then."


def test_personal_name_used_as_label_is_redacted():
    seen = []
    text = "Jordan: Hello there."
    result = process_with_speaker_labels(text, "prompt", make_fake_api(["Jordan"], seen), make_stats())
    assert result == "[REDACTED]: Hello there."


def test_role_label_is_restored():
    seen = []
    text = "Coach: Josh, how did it go?"
    result = process_with_speaker_labels(text, "prompt", make_fake_api(["Josh"], seen), make_stats())
    assert result == "Coach: [REDACTED], how did it go?"


def test_each_line_keeps_its_own_word_count():
    seen = []
    text = "Speaker A: One two three.\nSpeaker B: Four Josh.\nSpeaker A: Five."
    result = process_with_speaker_labels(text, "prompt", make_fake_api(["Josh"], seen), make_stats())
    assert result == "Speaker A: One two three.\nSpeaker B: Four [REDACTED].\nSpeaker A: Five."


def test_label_with_empty_body_is_kept():
    seen = []
    text = "Speaker A:\nSpeaker B: Hi Josh."
    result = process_with_speaker_labels(text, "prompt", make_fake_api(["Josh"], seen), make_stats())
    assert result == "Speaker A:\nSpeaker B: Hi [REDACTED]."


def test_text_without_labels_is_redacted_as_before():
    seen = []
    result = process_with_speaker_labels(
        "Josh went home.", "prompt", make_fake_api(["Josh"], seen), make_stats()
    )
    assert result == "[REDACTED] went home."
