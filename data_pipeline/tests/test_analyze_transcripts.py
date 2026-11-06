import sys
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_transcripts as at  # noqa: E402
import transcript_utils as tu  # noqa: E402

CFG = Namespace(labels=[], min_tokens=0, max_tokens_cap=600, fit_pct=90.0, clean_words_pct=10.0)

# Original: title line, then two turns. Timestamps are inline and must be ignored.
ORIGINAL = (
    "Lesson Fractions Day 1\n"
    "Coach: Welcome Maria 1 00:00:01.000 --> 00:00:02.000\n"
    "Teacher: Good morning Juan Perez today 2 00:00:02.000 --> 00:00:04.000\n"
)
# Human redacted Maria (per-word) and Juan Perez (per-word, two tokens).
HUMAN = (
    "Lesson Fractions Day 1\n"
    "Coach: Welcome [REDACTED] 1 00:00:01.000 --> 00:00:02.000\n"
    "Teacher: Good morning [REDACTED] [REDACTED] today 2 00:00:02.000 --> 00:00:04.000\n"
)
# LLM redacted Welcome (over-redaction), missed Maria (under-redaction),
# and collapsed Juan Perez into one token.
LLM = (
    "Lesson Fractions Day 1\n"
    "Coach: [REDACTED] Maria 1 00:00:01.000 --> 00:00:02.000\n"
    "Teacher: Good morning [REDACTED] today 2 00:00:02.000 --> 00:00:04.000\n"
)


class AnalyzeFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = {"original": ORIGINAL, "human": HUMAN, "llm": LLM}
        cls.result = at.analyze_file("t1.txt", texts, tu.label_regex(), CFG)

    def test_title_is_first_line_before_first_label(self):
        self.assertEqual(self.result["title"], "Lesson Fractions Day 1")

    def test_turns_exclude_timestamps(self):
        sizes = [t["size"] for t in self.result["turn_rows"]]
        self.assertEqual(sizes, [3, 6])  # "Coach: Welcome Maria", "Teacher: Good morning Juan Perez today"

    def test_over_redaction_is_llm_only_word(self):
        self.assertEqual([w["word"] for w in self.result["over_words"]], ["Welcome"])

    def test_under_redaction_is_human_only_word(self):
        self.assertEqual([w["word"] for w in self.result["under_words"]], ["Maria"])

    def test_multiword_style_per_word_for_human_and_collapsed_for_llm(self):
        styles = {(s["side"], s["example"]): s["style"] for s in self.result["spans"]}
        self.assertEqual(styles[("human", "Juan Perez")], "per-word")
        self.assertEqual(styles[("llm", "Juan Perez")], "collapsed")

    def test_word_rows_carry_file_name_for_tables(self):
        self.assertEqual(self.result["over_words"][0]["file"], "t1.txt")
        self.assertEqual(self.result["under_words"][0]["file"], "t1.txt")

    def test_no_run_crosses_a_turn(self):
        self.assertEqual(sum(self.result["crossing"].values()), 0)

    def test_no_nonliteral_redaction_tokens(self):
        self.assertFalse(any(self.result["variants"][v] for v in ("human", "llm")))
        self.assertFalse(any(self.result["glued"][v] for v in ("human", "llm")))


class HelperTests(unittest.TestCase):
    def test_redaction_runs(self):
        self.assertEqual(at.redaction_runs([False, True, True, False, True]), [(1, 3), (4, 5)])

    def test_percentile_interpolates(self):
        self.assertEqual(at.percentile([10, 20, 30, 40], 50), 25)
        self.assertIsNone(at.percentile([], 50))

    def test_word_shape_categories(self):
        self.assertEqual(at.word_shape("Maria"), "Capitalized")
        self.assertEqual(at.word_shape("USA"), "ALLCAPS")
        self.assertEqual(at.word_shape("the"), "stopword-like")
        self.assertEqual(at.word_shape("room5"), "contains digit")
        self.assertEqual(at.word_shape("table"), "lowercase/other")

    def test_extract_title_empty_when_no_preamble(self):
        self.assertEqual(at.extract_title("Coach: hi\n", tu.label_regex()), "")


if __name__ == "__main__":
    unittest.main()
