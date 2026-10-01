"""All prompt text lives here, separate from graph logic, so prompts can be
iterated on without touching orchestration code.
"""

from agent.config import TARGET_BRANCH_COVERAGE, TARGET_LINE_COVERAGE

# --------------------------------------------------------------------------
# 1. Porting: Python -> C++
# --------------------------------------------------------------------------

PORT_SYSTEM_PROMPT = """You are an expert C++ engineer. Convert Python code into high
performance, modern C++17. Respond only with semantically and syntactically correct C++
code. Follow RAII strictly (no raw new/delete without a clear ownership reason; prefer
smart pointers, containers, and stack allocation). Manage memory and resource lifetimes
correctly. Do not provide any explanation outside of occasional code comments. Do not
wrap your response in markdown code fences."""


def port_user_prompt(
    python_code: str,
    system_info: str,
    judge_feedback: str | None = None,
    extra_info: str | None = None,
) -> str:
    feedback_block = ""
    if judge_feedback:
        feedback_block = f"""
A previous attempt was reviewed by an automated code-quality judge and scored below the
passing threshold. Fix these specific issues in your next attempt:
{judge_feedback}
"""
    extra_block = ""
    if extra_info:
        extra_block = f"""
The human reviewer provided this additional guidance for this attempt -- follow it closely:
{extra_info}
"""
    return f"""Port this Python code to C++ with the most semantically and syntactically
accurate implementation that produces identical output in the least time.

System information:
{system_info}
{feedback_block}{extra_block}
Respond only with C++ code (no markdown fences, no explanation).

Python code to port:
```python
{python_code}
```"""


# --------------------------------------------------------------------------
# 2. LLM-as-judge: semantic relevance / RAII / memory management
# --------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a senior C++ code reviewer acting as an automated quality
gate. You will be given the original Python source and a candidate C++ port. Score the
port strictly and honestly. Respond ONLY with a single JSON object, no markdown, no
commentary outside the JSON."""


def judge_user_prompt(python_code: str, cpp_code: str) -> str:
    return f"""Evaluate the C++ port of this Python code on three dimensions, each scored
0-100:

1. "semantic_relevance": does the C++ code implement the same algorithm/behavior as the
   Python code, including edge cases (empty inputs, None/nullptr handling, off-by-one
   conditions, etc.)?
2. "raii_score": does the code follow RAII correctly (no leaked raw owning pointers,
   correct constructor/destructor pairing, exception-safety of resource acquisition)?
3. "memory_management_score": is memory managed correctly (no leaks, no use-after-free,
   no double-free, appropriate use of smart pointers/containers vs. raw pointers)?

Return exactly this JSON shape:
{{
  "semantic_relevance": <int 0-100>,
  "raii_score": <int 0-100>,
  "memory_management_score": <int 0-100>,
  "overall_score": <int 0-100, your holistic judgment, not necessarily the plain average>,
  "feedback": "<concise, actionable list of concrete issues to fix, or empty string if none>"
}}

Original Python code:
```python
{python_code}
```

Candidate C++ port:
```cpp
{cpp_code}
```"""


# --------------------------------------------------------------------------
# 3. Compile-error self-repair
# --------------------------------------------------------------------------

FIX_COMPILE_SYSTEM_PROMPT = """You are an expert C++ engineer. Fix the compiler/runtime
error in the given C++ code, which was ported from the given Python source. Respond only
with the complete corrected C++17 source code -- no markdown fences, no explanation."""


def fix_compile_user_prompt(
    python_code: str,
    cpp_code: str,
    command: list[str],
    stderr: str,
    returncode: int,
    extra_info: str | None = None,
) -> str:
    extra_block = f"\nAdditional guidance from the human reviewer:\n{extra_info}\n" if extra_info else ""
    return f"""The following C++ code, ported from the Python source below, failed to
compile or run.

Command: {command}
Return code: {returncode}
Error output:
{stderr}
{extra_block}
Original Python source (for reference -- preserve its behavior):
```python
{python_code}
```

Current C++ source:
```cpp
{cpp_code}
```

Return the complete, fixed C++17 source code only."""


# --------------------------------------------------------------------------
# 4. Test generation (GoogleTest)
# --------------------------------------------------------------------------

TEST_SYSTEM_PROMPT = f"""You are an expert C++ software engineer and Google Test developer.
Generate Google Test unit tests for the given C++17 source code.

Requirements:
- Respond ONLY with valid JSON: {{"test_code": "...", "coverage_justification": "..."}}
- No markdown or code fences inside or outside the JSON.
- Never duplicate the implementation in the test file; #include the implementation file
  (with main() stripped) at the top of the test file instead, using the exact relative
  path given to you in the user prompt.
- Target {TARGET_LINE_COVERAGE}% line coverage and {TARGET_BRANCH_COVERAGE}%+ branch
  coverage. Cover normal cases, edge cases, invalid inputs, and boundary conditions.
- "coverage_justification": if you believe some lines/branches are genuinely untestable
  or unreachable (e.g. defensive code, platform-specific branches) and the targets above
  cannot realistically be met, explain exactly which parts and why. Leave this an empty
  string if you believe the targets are fully achievable by your generated tests.
- Do not modify the implementation.
- Do not generate a main() function (the test binary links against gtest_main)."""


def test_user_prompt(cpp_code: str, include_path: str = "main_test.cpp") -> str:
    return f"""Generate Google Test unit tests for this C++17 code.

The implementation (with main() stripped) is available at a file you must include via:
#include "{include_path}"
at the top of your test file -- use exactly that path (it accounts for the project's
directory layout, where the test file and the implementation file live in different
folders). Do not duplicate the implementation.

Return exactly:
{{
  "test_code": "<complete C++ Google Test source>",
  "coverage_justification": "<empty string, or an explanation if targets can't be met>"
}}

C++ source:
```cpp
{cpp_code}
```"""


# --------------------------------------------------------------------------
# 5. Test compile/run-error self-repair
# --------------------------------------------------------------------------

FIX_TEST_SYSTEM_PROMPT = """You are an expert C++ software engineer and Google Test
developer. A generated Google Test file failed to compile or run against a C++
implementation. Your first job is diagnosis, not just repair: a test failure can mean
either (a) the test itself is wrong (bad assertion, wrong include, API mismatch), or
(b) the test is correct and has caught a genuine bug in the implementation it's testing.
Look at the actual assertion/compiler output before deciding.

Requirements:
- Respond ONLY with valid JSON:
  {"diagnosis": "test" | "implementation",
   "test_code": "<complete test source -- fixed if diagnosis is 'test', otherwise the
                  original test source unchanged>",
   "implementation_code": "<complete corrected implementation, ONLY if diagnosis is
                            'implementation'; omit this key entirely if diagnosis is 'test'>",
   "explanation": "<one or two sentences on what was actually wrong>"}
- If diagnosis is "implementation": fix the implementation to match the behavior the
  original Python code specifies (it will be recompiled and the same tests re-run
  against it). Do not weaken or delete the test to make it pass.
  CRITICAL: "implementation_code" must be the COMPLETE file, including an intact
  `int main(...)` function -- this file is compiled on its own as a standalone program,
  separately from the test binary, so it must remain independently compilable and
  runnable. Do not drop or omit main() just because the failing tests link against a
  main()-stripped copy of this file; the original file with main() must still exist.
- If diagnosis is "test": fix the test; do not modify the implementation.
- Keep the #include of the implementation file exactly as it appears in the current
  test source -- do not change that path.
- Do not generate a main() function in the test file."""


def fix_test_user_prompt(
    test_code: str,
    cpp_code: str,
    python_code: str,
    command: list[str],
    stderr: str,
    returncode: int,
    extra_info: str | None = None,
) -> str:
    extra_block = f"\nAdditional guidance for this attempt -- follow it closely:\n{extra_info}\n" if extra_info else ""
    return f"""The following Google Test file failed against the given implementation.

Command: {command}
Return code: {returncode}
Error output:
{stderr}
{extra_block}
Original Python source the implementation was ported from (the ground truth for
expected behavior):
```python
{python_code}
```

Current C++ implementation under test:
```cpp
{cpp_code}
```

Current test source:
```cpp
{test_code}
```

Diagnose whether the test or the implementation is at fault, then return the JSON
described in your instructions."""
