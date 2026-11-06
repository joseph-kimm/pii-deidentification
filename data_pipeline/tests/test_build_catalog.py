import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_catalog as bc  # noqa: E402

ORIGINAL = (
    "Teacher: welcome to the class 1 00:00:01.000 --> 00:00:03.000\n"
    "Student: hi there 2 00:00:03.000 --> 00:00:04.000\n"
    "Teacher: Maria Lopez will present today 3 00:00:04.000 --> 00:00:07.000\n"
)
HUMAN = (
    "Teacher: welcome to the class 1 00:00:01.000 --> 00:00:03.000\n"
    "Student: hi there 2 00:00:03.000 --> 00:00:04.000\n"
    "Teacher: [REDACTED] [REDACTED] will present today 3 00:00:04.000 --> 00:00:07.000\n"
)
LLM = (
    "Teacher: welcome to the class 1 00:00:01.000 --> 00:00:03.000\n"
    "Student: hi there 2 00:00:03.000 --> 00:00:04.000\n"
    "Teacher: [REDACTED] will present today 3 00:00:04.000 --> 00:00:07.000\n"
)


class BuildCatalogEndToEndTests(unittest.TestCase):
    def run_build(self, labels_arg=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, text in (("original", ORIGINAL), ("human", HUMAN), ("llm", LLM)):
                (root / folder).mkdir()
                (root / folder / "t1.txt").write_text(text, encoding="utf-8")
            argv = ["--original", str(root / "original"), "--human", str(root / "human"),
                    "--llm", str(root / "llm"), "--out", str(root / "catalog.csv"),
                    "--min-tokens", "0", "--max-tokens", "600", "--tokenizer", "words"]
            if labels_arg:
                argv += ["--labels", labels_arg]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                bc.build(bc.parse_args(argv))
            with open(root / "catalog.csv", newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f)), err.getvalue()

    def test_timestamps_never_counted_as_words(self):
        rows, _ = self.run_build(labels_arg="Student")
        # Chunk size is the word count of the turn (label token included), with
        # timestamp tokens already stripped, so it is never the raw position span.
        sizes = {r["turn_idx"]: int(r["size"]) for r in rows}
        self.assertEqual(sizes, {"0": 5, "1": 3, "2": 6})

    def test_word_total_is_speech_plus_labels_only(self):
        rows, _ = self.run_build(labels_arg="Student")
        # 5 + 3 + 6 words; timestamp tokens (index, time, arrow, time) excluded.
        self.assertEqual(sum(int(r["size"]) for r in rows), 14)

    def test_default_labels_leave_student_line_in_teacher_turn(self):
        rows, err = self.run_build()
        sizes = {r["turn_idx"]: int(r["size"]) for r in rows}
        self.assertEqual(sizes, {"0": 8, "1": 6})
        self.assertNotIn("no speaker labels", err)

    def test_extra_labels_split_turns(self):
        rows, _ = self.run_build(labels_arg="Student")
        self.assertEqual(len({r["turn_idx"] for r in rows}), 3)


if __name__ == "__main__":
    unittest.main()
