import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcript_utils import TIMESTAMP_RE, label_regex, strip_timestamps  # noqa: E402


class StripTimestampsTests(unittest.TestCase):
    def test_removes_inline_index_and_timestamp_keeping_speech(self):
        text = "Teacher: yeah 11 00:01:12.000 --> 00:01:13.153\n"
        self.assertEqual(strip_timestamps(text), "Teacher: yeah \n")

    def test_removes_timestamp_at_start_of_line_without_index(self):
        text = "Coach: 00:00:03.5 --> 00:00:09.25 Let's begin.\n"
        self.assertEqual(strip_timestamps(text), "Coach:  Let's begin.\n")

    def test_handles_short_minute_second_times(self):
        text = "Speaker 1: Hi. 12 0:03 --> 0:09\n"
        self.assertEqual(strip_timestamps(text), "Speaker 1: Hi. \n")

    def test_keeps_bare_number_in_speech(self):
        text = "Speaker 2: I counted to 42 and stopped.\n"
        self.assertEqual(strip_timestamps(text), text)

    def test_keeps_arrow_that_is_not_a_timestamp(self):
        text = "Teacher: go --> there.\n"
        self.assertEqual(strip_timestamps(text), text)

    def test_keeps_bare_time_in_speech(self):
        text = "Coach: meet at 10:30 today.\n"
        self.assertEqual(strip_timestamps(text), text)

    def test_leaves_text_without_timestamps_unchanged(self):
        text = "Teacher: Plain.\n\nSpeaker A: Also plain.\n"
        self.assertEqual(strip_timestamps(text), text)

    def test_keeps_newlines_so_line_structure_is_preserved(self):
        text = "Teacher: a 1 00:00:01 --> 00:00:02\nCoach: b 2 00:00:02 --> 00:00:03\n"
        self.assertEqual(strip_timestamps(text).count("\n"), 2)

    def test_timestamp_regex_does_not_match_plain_text(self):
        self.assertIsNone(TIMESTAMP_RE.search("no times here 12 at all"))


class LabelRegexTests(unittest.TestCase):
    def setUp(self):
        self.rx = label_regex()

    def test_matches_default_roles_at_line_start(self):
        text = "Teacher: a\nCoach: b\nSpeaker 1: c\nSpeaker A: d\n"
        self.assertEqual(len(self.rx.findall(text)), 4)

    def test_does_not_match_label_inside_a_line(self):
        self.assertIsNone(self.rx.search("She said Teacher: hello"))

    def test_extra_roles_are_added(self):
        rx = label_regex(["Student"])
        self.assertIsNotNone(rx.search("Student: hi\n"))
        self.assertIsNone(self.rx.search("Student: hi\n"))

    def test_speaker_family_needs_colon(self):
        self.assertIsNone(self.rx.search("Speaker 1 said nothing\n"))


if __name__ == "__main__":
    unittest.main()
