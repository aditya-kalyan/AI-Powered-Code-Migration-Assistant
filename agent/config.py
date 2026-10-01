"""Central configuration constants for the migration agent.

Keeping these in one place makes the pipeline's thresholds and limits easy to
audit/tune without hunting through node logic.
"""

import platform

# --- LLM ---
DEFAULT_MODEL = "gpt-4o-mini"

# --- Retry / escalation policy ---
# Each gated stage (judge, compile, test-compile) auto-retries up to this many
# times before escalating to a human-in-the-loop (HITL) checkpoint. After HITL
# fires, exactly ONE more attempt is made (either an LLM retry seeded with the
# user's extra context, or the user's own hand-edited code) and the pipeline
# then moves on regardless of the outcome -- no infinite HITL loops.
MAX_AUTO_RETRIES = 3

# --- LLM-as-judge gate (semantic relevance / RAII / memory management) ---
JUDGE_PASS_THRESHOLD = 80  # out of 100

# --- Test generation coverage targets (aspirational; requested of the LLM) ---
TARGET_LINE_COVERAGE = 100      # percent
TARGET_BRANCH_COVERAGE = 90     # percent

# --- Platform helpers ---
IS_WINDOWS = platform.system() == "Windows"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""
RUN_PREFIX = "" if IS_WINDOWS else "./"

# Required system tools. All three are checked up front; the pipeline degrades
# gracefully (rather than failing outright) depending on which subset is present.
REQUIRED_TOOLS = ("g++", "gtest", "cppcheck")

SUBPROCESS_TIMEOUT_SECONDS = 60

# --- Output layout ---
# All generated artifacts land under these three subfolders of the *current
# working directory* (i.e. wherever `python app.py` was launched from) -- not
# a system temp directory. Re-running overwrites the previous run's files.
SRC_DIR_NAME = "src"
TEST_DIR_NAME = "tests"
REPORTS_DIR_NAME = "reports"
