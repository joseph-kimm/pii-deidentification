"""
Shared transcript helpers: speaker labels and inline timestamps.

Speaker labels: a line that starts with a role name followed by a colon.
Default roles are "Coach", "Teacher" and "Speaker X" (X = 1-3 letters/digits).
Extra roles can be added with --labels in build_catalog.py / analyze_transcripts.py.

Timestamps: an optional index, then "start --> end" in H:MM:SS.mmm or M:SS form,
e.g. "yeah 11 00:01:12.000 --> 00:01:13.153". The match can sit anywhere in a
line, so it is removed from the text itself and the speech before it is kept.
Only the "-->" signature is matched, so a bare time in speech is left alone.
"""
import re

DEFAULT_ROLES = ("Coach", "Teacher")
SPEAKER_FAMILY = r"Speaker\s*[A-Za-z0-9]{1,3}"
TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?"
TIMESTAMP_RE = re.compile(rf"(?:\b\d+[ \t]+)?{TIME}[ \t]*-->[ \t]*{TIME}")


def label_regex(extra_roles=()):
    """Compiled regex matching a speaker label at the start of a line."""
    roles = [re.escape(role.strip()) for role in (*DEFAULT_ROLES, *extra_roles) if role.strip()]
    alternatives = "|".join([*roles, SPEAKER_FAMILY])
    return re.compile(rf"^[ \t]*(?:{alternatives})[ \t]*:", re.M)


def strip_timestamps(text):
    """Remove inline 'index start --> end' timestamps, keeping the rest of each line."""
    return TIMESTAMP_RE.sub("", text)
