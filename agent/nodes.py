"""Every node (and routing function) in the migration graph.

Naming convention: a handful of stages are gated by a bounded auto-retry loop
that escalates to a human-in-the-loop (HITL) checkpoint after
`MAX_AUTO_RETRIES` failures. For each such stage there are up to five nodes:

    <stage>                 the action itself (port / compile / test-compile)
    fix_<stage>              LLM repair, looped back into <stage>
    human_review_<stage>     the interrupt() checkpoint
    <stage>_final            same action as <stage>, reused after HITL resolves,
                              wired with an UNCONDITIONAL outgoing edge (no more
                              looping -- the pipeline moves on either way)
    apply_<stage>_edit       writes the human's hand-edited code directly,
                              skipping the LLM, before <stage>_final re-validates it

This keeps the "3 auto-retries, then exactly one human-guided attempt, then
move on regardless" policy identical across all three gates (judge, compile,
test-compile) without copy-pasting graph wiring logic.
"""

from __future__ import annotations

import os
from typing import Any

from langgraph.types import interrupt

from agent.config import (
    JUDGE_PASS_THRESHOLD,
    MAX_AUTO_RETRIES,
    REPORTS_DIR_NAME,
    SRC_DIR_NAME,
    TEST_DIR_NAME,
)
from agent.llm_client import extract_json, get_client, strip_code_fences
from agent.prompts import (
    FIX_COMPILE_SYSTEM_PROMPT,
    FIX_TEST_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PORT_SYSTEM_PROMPT,
    TEST_SYSTEM_PROMPT,
    fix_compile_user_prompt,
    fix_test_user_prompt,
    judge_user_prompt,
    port_user_prompt,
    test_user_prompt,
)
from agent.state import MigrationState
from agent.system_probe import (
    classify_tool_mode,
    collect_system_info,
    default_compile_command,
    default_coverage_html_command,
    default_coverage_text_command,
    default_run_command,
    default_test_compile_command,
    default_test_run_command,
    has_main_function,
    install_steps_for,
    parse_gcovr_summary,
    probe_tools,
    remove_main_function,
    run_cmd,
)


def _write(workdir: str, filename: str, content: str) -> None:
    with open(os.path.join(workdir, filename), "w", encoding="utf-8") as f:
        f.write(content)


# ==========================================================================
# Stage 0: tool detection
# ==========================================================================

def check_tools(state: MigrationState) -> dict:
    tool_status = probe_tools()
    mode = classify_tool_mode(tool_status)
    missing = [name for name, present in tool_status.items() if not present]
    install_steps = install_steps_for(missing) if missing else []
    system_info = collect_system_info()

    # Every artifact this run produces lands under `state["workdir"]` (which
    # the UI sets to the current working directory, not a system temp dir --
    # see ui/blocks.py):
    #   <workdir>/src/            -- SRC_DIR_NAME
    #   <workdir>/tests/          -- TEST_DIR_NAME
    #   <workdir>/tests/reports/  -- REPORTS_DIR_NAME, nested INSIDE tests/
    #
    # reports_dir is deliberately built as a second os.path.join on top of
    # test_dir (not a single compound string like "tests/reports" joined
    # straight onto workdir). A compound string's nesting depends on the OS
    # interpreting its separator character correctly -- e.g. "tests\reports"
    # nests fine on Windows but becomes one oddly-named sibling directory on
    # POSIX, since backslash isn't a path separator there. Two plain-leaf-name
    # joins can't have that failure mode.
    workdir = state["workdir"]

    def _subdir(base: str, name: str) -> str:
        """os.path.join silently discards `base` if `name` starts with a
        path separator (it treats that as an absolute path reset) -- e.g.
        os.path.join('/a/b', '/c/d') == '/c/d', not '/a/b/c/d'. Stripping any
        leading separator here means a misconfigured dir-name constant (with
        a stray leading '/' or '\\') degrades to "just nested under base"
        instead of silently writing outside the project folder."""
        return os.path.join(base, name.lstrip("/\\"))

    src_dir = _subdir(workdir, SRC_DIR_NAME)
    test_dir = _subdir(workdir, TEST_DIR_NAME)
    reports_dir = _subdir(test_dir, REPORTS_DIR_NAME)  # nested INSIDE tests/, not a sibling
    for d in (src_dir, test_dir, reports_dir):
        os.makedirs(d, exist_ok=True)

    logs = [f"[check_tools] g++={tool_status['g++']} gtest={tool_status['gtest']} "
            f"cppcheck={tool_status['cppcheck']} -> mode={mode}",
            f"[check_tools] Output folders: {src_dir}, {test_dir}, {reports_dir}"]
    if mode == "none":
        logs.append("[check_tools] No required tools found. Halting before code generation.")
    elif mode == "partial":
        logs.append(f"[check_tools] Missing: {missing}. Will generate everything an LLM can "
                     f"produce, and skip local execution steps that need the missing tool(s).")

    return {
        "tool_status": tool_status,
        "tool_mode": mode,
        "install_steps": install_steps,
        "system_info": system_info,
        "src_dir": src_dir,
        "test_dir": test_dir,
        "reports_dir": reports_dir,
        "logs": logs,
        "status": "tools_checked",
    }


def route_after_tool_check(state: MigrationState) -> str:
    return "halt" if state["tool_mode"] == "none" else "proceed"


# ==========================================================================
# Stage 1: porting + LLM-as-judge gate
# ==========================================================================

def port_to_cpp(state: MigrationState) -> dict:
    """Ports Python -> C++. Reused for both the first attempt and every
    auto-retry: if a `judge_result` already exists in state (i.e. this is a
    retry), its feedback is folded into the prompt."""
    client = get_client()
    judge_feedback = None
    is_retry = "judge_result" in state
    if is_retry:
        judge_feedback = state["judge_result"].get("feedback") or None

    resp = client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": PORT_SYSTEM_PROMPT},
            {"role": "user", "content": port_user_prompt(
                state["python_code"], state["system_info"], judge_feedback=judge_feedback
            )},
        ],
    )
    cpp_code = strip_code_fences(resp.choices[0].message.content)

    src_dir = state["src_dir"]
    _write(src_dir, "main.cpp", cpp_code)
    cpp_test_impl = remove_main_function(cpp_code)
    _write(src_dir, "main_test.cpp", cpp_test_impl)

    result: dict[str, Any] = {
        "cpp_full": cpp_code,
        "cpp_test_impl": cpp_test_impl,
        "logs": [f"[port_to_cpp] {'Retry: regenerated' if is_retry else 'Wrote'} "
                 f"{src_dir}/main.cpp ({len(cpp_code)} chars)."],
        "status": "ported",
    }
    if is_retry:
        result["judge_retry_count"] = state.get("judge_retry_count", 0) + 1
    return result


def port_to_cpp_extra(state: MigrationState) -> dict:
    """The single extra LLM attempt after judge HITL escalation, seeded with
    the human's free-text guidance."""
    client = get_client()
    extra_info = state.get("judge_extra_info")
    resp = client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": PORT_SYSTEM_PROMPT},
            {"role": "user", "content": port_user_prompt(
                state["python_code"], state["system_info"],
                judge_feedback=state.get("judge_result", {}).get("feedback"),
                extra_info=extra_info,
            )},
        ],
    )
    cpp_code = strip_code_fences(resp.choices[0].message.content)
    src_dir = state["src_dir"]
    _write(src_dir, "main.cpp", cpp_code)
    cpp_test_impl = remove_main_function(cpp_code)
    _write(src_dir, "main_test.cpp", cpp_test_impl)

    return {
        "cpp_full": cpp_code,
        "cpp_test_impl": cpp_test_impl,
        "logs": ["[port_to_cpp_extra] HITL-guided retry: regenerated main.cpp with human "
                 "guidance."],
        "status": "ported_after_hitl",
    }


def apply_judge_edit(state: MigrationState) -> dict:
    """Human chose to hand-edit the C++ directly instead of another LLM attempt."""
    edited = state.get("judge_edited_code", state.get("cpp_full", ""))
    src_dir = state["src_dir"]
    _write(src_dir, "main.cpp", edited)
    cpp_test_impl = remove_main_function(edited)
    _write(src_dir, "main_test.cpp", cpp_test_impl)
    return {
        "cpp_full": edited,
        "cpp_test_impl": cpp_test_impl,
        "logs": ["[apply_judge_edit] Using human-edited C++ code, skipping further judging."],
        "status": "judge_edit_applied",
    }


def judge_cpp(state: MigrationState) -> dict:
    """LLM-as-judge: scores semantic relevance, RAII compliance, and memory
    management. This function is reused, unchanged, for both the gating call
    (`judge_cpp`) and the post-HITL reporting-only call (`judge_cpp_final`)."""
    client = get_client()
    resp = client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_user_prompt(state["python_code"], state["cpp_full"])},
        ],
        response_format={"type": "json_object"},
    )
    judge_result = extract_json(resp.choices[0].message.content)
    score = judge_result.get("overall_score", 0)
    return {
        "judge_result": judge_result,
        "logs": [f"[judge_cpp] overall_score={score} "
                 f"(semantic={judge_result.get('semantic_relevance')}, "
                 f"raii={judge_result.get('raii_score')}, "
                 f"memory={judge_result.get('memory_management_score')})"],
        "status": "judged",
    }


def route_after_judge(state: MigrationState) -> str:
    score = state["judge_result"].get("overall_score", 0)
    threshold = state.get("judge_threshold", JUDGE_PASS_THRESHOLD)
    if score >= threshold:
        return "pass"
    if state.get("judge_retry_count", 0) < state.get("max_retries", MAX_AUTO_RETRIES):
        return "retry"
    return "hitl"


def human_review_judge(state: MigrationState) -> dict:
    decision = interrupt({
        "stage": "judge",
        "message": "The LLM-as-judge could not get the C++ port above the quality "
                    "threshold after automatic retries. Provide extra guidance for one "
                    "more automated attempt, or edit the C++ directly.",
        "cpp_code": state["cpp_full"],
        "judge_result": state["judge_result"],
    })
    action = decision.get("action", "retry_with_info")
    result: dict[str, Any] = {
        "judge_decision": action,
        "logs": [f"[human_review_judge] User chose: {action}"],
    }
    if action == "retry_with_info":
        result["judge_extra_info"] = decision.get("extra_info", "")
    else:
        result["judge_edited_code"] = decision.get("edited_code", state["cpp_full"])
    return result


def route_after_judge_hitl(state: MigrationState) -> str:
    return "extra_retry" if state.get("judge_decision") == "retry_with_info" else "edited"


# ==========================================================================
# Stage 2: compile gate (only runs if g++ is available)
# ==========================================================================

def compile_gate(state: MigrationState) -> dict:
    """Builds the compile/run commands. Actual compilation is skipped here if
    g++ isn't available -- the routing function decides whether to invoke the
    real compile node or bypass it."""
    src_dir = state["src_dir"]
    compile_command = default_compile_command(src_dir, "main.cpp", "main")
    run_command = default_run_command(src_dir, "main")
    return {
        "compile_command": compile_command,
        "run_command": run_command,
        "logs": ["[compile_gate] Prepared compile/run commands."]
                if state["tool_status"]["g++"]
                else ["[compile_gate] g++ not found -- skipping compilation; "
                      "main.cpp will still be delivered as source."],
        "status": "compile_gate_checked",
    }


def route_after_compile_gate(state: MigrationState) -> str:
    return "compile" if state["tool_status"]["g++"] else "skip"


def compile_and_run_cpp(state: MigrationState) -> dict:
    """Compiles and runs main.cpp. Reused unchanged as `compile_cpp_final`
    after HITL resolves the compile-error gate."""
    src_dir = state["src_dir"]
    compile_result = run_cmd(state["compile_command"], cwd=src_dir)
    if not compile_result["success"]:
        return {
            "compile_result": {"stage": "compile", **compile_result},
            "logs": [f"[compile_and_run_cpp] Compile FAILED (rc={compile_result['returncode']})."],
            "status": "cpp_compile_failed",
        }
    run_result = run_cmd(state["run_command"], cwd=src_dir)
    stage = "done" if run_result["success"] else "run"
    return {
        "compile_result": {"stage": stage, **run_result},
        "logs": [f"[compile_and_run_cpp] Compile OK. Run "
                 f"{'succeeded' if run_result['success'] else 'FAILED'}."],
        "status": "cpp_run_done" if run_result["success"] else "cpp_run_failed",
    }


def route_after_compile(state: MigrationState) -> str:
    if state["compile_result"].get("success"):
        return "success"
    if state.get("compile_retry_count", 0) < state.get("max_retries", MAX_AUTO_RETRIES):
        return "retry"
    return "hitl"


def fix_cpp_error(state: MigrationState) -> dict:
    """LLM repair of a compile/run failure. Reused unchanged as
    `fix_cpp_error_extra` for the post-HITL single guided retry -- the only
    difference is whether `compile_extra_info` is present in state."""
    client = get_client()
    failure = state["compile_result"]
    extra_info = state.get("compile_extra_info")
    resp = client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": FIX_COMPILE_SYSTEM_PROMPT},
            {"role": "user", "content": fix_compile_user_prompt(
                state["python_code"], state["cpp_full"], failure["command"],
                failure["stderr"], failure["returncode"], extra_info=extra_info,
            )},
        ],
    )
    fixed_cpp = strip_code_fences(resp.choices[0].message.content)
    src_dir = state["src_dir"]
    _write(src_dir, "main.cpp", fixed_cpp)
    cpp_test_impl = remove_main_function(fixed_cpp)
    _write(src_dir, "main_test.cpp", cpp_test_impl)

    result: dict[str, Any] = {
        "cpp_full": fixed_cpp,
        "cpp_test_impl": cpp_test_impl,
        "logs": [f"[fix_cpp_error] {'HITL-guided' if extra_info else 'Auto'} retry: "
                 f"regenerated main.cpp."],
        "status": "cpp_fix_attempted",
    }
    if not extra_info:
        result["compile_retry_count"] = state.get("compile_retry_count", 0) + 1
    return result


def human_review_compile(state: MigrationState) -> dict:
    decision = interrupt({
        "stage": "compile",
        "message": "The C++ port keeps failing to compile/run after automatic retries. "
                    "Provide extra guidance for one more automated attempt, or edit the "
                    "C++ directly.",
        "cpp_code": state["cpp_full"],
        "compile_result": state["compile_result"],
    })
    action = decision.get("action", "retry_with_info")
    result: dict[str, Any] = {
        "compile_decision": action,
        "logs": [f"[human_review_compile] User chose: {action}"],
    }
    if action == "retry_with_info":
        result["compile_extra_info"] = decision.get("extra_info", "")
    else:
        result["compile_edited_code"] = decision.get("edited_code", state["cpp_full"])
    return result


def route_after_compile_hitl(state: MigrationState) -> str:
    return "extra_retry" if state.get("compile_decision") == "retry_with_info" else "edited"


def apply_compile_edit(state: MigrationState) -> dict:
    edited = state.get("compile_edited_code", state.get("cpp_full", ""))
    src_dir = state["src_dir"]
    _write(src_dir, "main.cpp", edited)
    cpp_test_impl = remove_main_function(edited)
    _write(src_dir, "main_test.cpp", cpp_test_impl)
    return {
        "cpp_full": edited,
        "cpp_test_impl": cpp_test_impl,
        "logs": ["[apply_compile_edit] Using human-edited C++ code; recompiling once more."],
        "status": "compile_edit_applied",
    }


# ==========================================================================
# Stage 3: static analysis (cppcheck) -- informational, non-blocking
# ==========================================================================

def cppcheck_gate(state: MigrationState) -> str:
    return "run" if state["tool_status"]["cppcheck"] else "skip"


def cppcheck_analysis(state: MigrationState) -> dict:
    result = run_cmd(["cppcheck", "--enable=all", "--inconclusive", "--std=c++17", "main.cpp"],
                      cwd=state["src_dir"], timeout=30)
    # cppcheck writes its findings to stderr by default.
    findings = result["stderr"]
    error_count = findings.count("error:")
    warning_count = findings.count("warning:")
    _write(state["reports_dir"], "cppcheck.txt", findings or "(no findings)")
    return {
        "cppcheck_result": {
            "ran": True,
            "raw": findings,
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "logs": [f"[cppcheck_analysis] {error_count} error(s), {warning_count} warning(s) "
                 f"(informational -- not blocking the pipeline)."],
        "status": "cppcheck_done",
    }


def skip_cppcheck(state: MigrationState) -> dict:
    return {
        "cppcheck_result": {"ran": False},
        "logs": ["[cppcheck_analysis] cppcheck not found -- skipped static analysis."],
        "status": "cppcheck_skipped",
    }


# ==========================================================================
# Stage 4: test generation (always runs -- LLM-only, no local tool needed)
# ==========================================================================

def generate_tests(state: MigrationState) -> dict:
    client = get_client()
    # test.cpp lives in <workdir>/tests/, and #includes the stripped
    # implementation from <workdir>/src/main_test.cpp -- tell the LLM the
    # exact relative include path for this layout.
    include_path = f"../{SRC_DIR_NAME}/main_test.cpp"
    resp = client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": TEST_SYSTEM_PROMPT},
            {"role": "user", "content": test_user_prompt(state["cpp_full"], include_path=include_path)},
        ],
        response_format={"type": "json_object"},
    )
    parsed = extract_json(resp.choices[0].message.content)
    test_code = parsed["test_code"]
    justification = parsed.get("coverage_justification", "")

    test_dir = state["test_dir"]
    workdir = state["workdir"]
    _write(test_dir, "test.cpp", test_code)

    # Coverage commands are run with cwd=workdir (the project root, parent of
    # both src/ and tests/) rather than cwd=test_dir -- see
    # `system_probe.default_coverage_text_command` for exactly why running
    # them rooted inside tests/ silently drops all coverage data.
    test_dir_rel = os.path.relpath(test_dir, workdir).replace(os.sep, "/")
    html_output_rel = os.path.relpath(
        os.path.join(state["reports_dir"], "index.html"), start=workdir
    ).replace(os.sep, "/")

    return {
        "test_code": test_code,
        "test_coverage_justification": justification,
        "test_compile_command": default_test_compile_command(test_dir),
        "test_run_command": default_test_run_command(test_dir),
        "coverage_command": default_coverage_text_command(test_dir_rel),
        "coverage_html_command": default_coverage_html_command(test_dir_rel, html_output_rel),
        "logs": [f"[generate_tests] Wrote {test_dir}/test.cpp ({len(test_code)} chars)."
                 + (f" LLM coverage note: {justification}" if justification else "")],
        "status": "tests_generated",
    }


# ==========================================================================
# Stage 5: test compile/run gate (only runs if g++ AND gtest are available,
# and only if the main implementation actually compiled)
# ==========================================================================

def test_gate(state: MigrationState) -> str:
    can_run = state["tool_status"]["g++"] and state["tool_status"]["gtest"]
    compiled_ok = state.get("compile_result", {}).get("success", False)
    return "compile" if (can_run and compiled_ok) else "skip"


def compile_and_run_tests(state: MigrationState) -> dict:
    """Compiles + runs test.cpp, then attempts a coverage report. Reused
    unchanged as `compile_and_run_tests_final` after test-stage HITL.

    Always re-syncs (recompiles) main.cpp first. This matters because
    `fix_test_error` can decide the *implementation* was the actual bug and
    patch main.cpp/main_test.cpp instead of test.cpp -- without this resync,
    that patched implementation would never actually get rebuilt before the
    tests are re-run against it.
    """
    src_dir = state["src_dir"]
    test_dir = state["test_dir"]

    resync = run_cmd(state["compile_command"], cwd=src_dir)
    if not resync["success"]:
        return {
            "test_result": {"stage": "main_resync", **resync},
            "logs": [f"[compile_and_run_tests] main.cpp failed to recompile "
                     f"(rc={resync['returncode']}) -- an implementation fix likely "
                     f"introduced a new compile error."],
            "status": "main_resync_failed",
        }

    compile_result = run_cmd(state["test_compile_command"], cwd=test_dir)
    if not compile_result["success"]:
        return {
            "test_result": {"stage": "compile", **compile_result},
            "logs": [f"[compile_and_run_tests] Test compile FAILED (rc={compile_result['returncode']})."],
            "status": "test_compile_failed",
        }

    run_result = run_cmd(state["test_run_command"], cwd=test_dir)
    if not run_result["success"]:
        return {
            "test_result": {"stage": "run", **run_result},
            "logs": ["[compile_and_run_tests] Test run FAILED."],
            "status": "test_run_failed",
        }

    coverage_report: dict[str, Any] = {"available": False}
    workdir = state["workdir"]
    cov = run_cmd(state["coverage_command"], cwd=workdir, timeout=30)
    if cov["success"]:
        coverage_report = {"available": True, "raw": cov["stdout"], **parse_gcovr_summary(cov["stdout"])}
        # Also render the HTML report into <workdir>/tests/reports/index.html.
        # Kept as a best-effort second call -- if it fails, the text summary
        # above is still reported.
        html_cmd = state.get("coverage_html_command")
        if html_cmd:
            html_result = run_cmd(html_cmd, cwd=workdir, timeout=30)
            coverage_report["html_report_path"] = os.path.join(state["reports_dir"], "index.html")
            coverage_report["html_generated"] = html_result["success"]
            if not html_result["success"]:
                coverage_report["html_error"] = html_result["stderr"][:500]
    else:
        coverage_report = {"available": False, "note": "gcovr not found or failed; install it for a coverage report.",
                            "raw_stderr": cov["stderr"][:500]}

    return {
        "test_result": {"stage": "done", "success": True, "compile": compile_result, "run": run_result},
        "coverage_report": coverage_report,
        "logs": [f"[compile_and_run_tests] Tests PASSED. "
                 f"{'Coverage: ' + str(coverage_report) if coverage_report.get('available') else coverage_report.get('note', '')}"],
        "status": "tests_passed",
    }


def route_after_test_compile(state: MigrationState) -> str:
    if state["test_result"].get("success"):
        return "success"
    if state.get("test_retry_count", 0) < state.get("max_retries", MAX_AUTO_RETRIES):
        return "retry"
    return "hitl"


def fix_test_error(state: MigrationState) -> dict:
    """Reused unchanged as `fix_test_error_extra` for the post-HITL guided retry.

    Unlike a naive "always regenerate test.cpp" repair, this gives the LLM
    both the test and the implementation and asks it to diagnose which one is
    actually at fault -- see `agent.prompts.FIX_TEST_SYSTEM_PROMPT`. If the
    implementation is the culprit, main.cpp/main_test.cpp get patched too;
    `compile_and_run_tests` always re-syncs (recompiles) main.cpp before the
    next test run, so that patch actually takes effect.
    """
    client = get_client()
    failure = state["test_result"]
    extra_info = state.get("test_extra_info")
    resp = client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": FIX_TEST_SYSTEM_PROMPT},
            {"role": "user", "content": fix_test_user_prompt(
                state["test_code"], state["cpp_full"], state["python_code"],
                failure["command"], failure["stderr"], failure["returncode"],
                extra_info=extra_info,
            )},
        ],
        response_format={"type": "json_object"},
    )
    parsed = extract_json(resp.choices[0].message.content)
    diagnosis = parsed.get("diagnosis", "test")
    fixed_test = parsed.get("test_code", state["test_code"])
    _write(state["test_dir"], "test.cpp", fixed_test)

    result: dict[str, Any] = {"test_code": fixed_test}
    logs = [f"[fix_test_error] Diagnosis: {diagnosis}. {parsed.get('explanation', '')}"]

    if diagnosis == "implementation" and parsed.get("implementation_code"):
        fixed_cpp = strip_code_fences(parsed["implementation_code"])
        if not has_main_function(fixed_cpp):
            # The model's "corrected implementation" dropped main() -- likely
            # because it was focused on making the tests (which link against
            # the main()-stripped main_test.cpp) pass, and didn't realize
            # main.cpp must remain independently compilable. Writing this
            # would silently break main.cpp (surfaces later as an opaque
            # "undefined reference to WinMain" linker error on MinGW). Refuse
            # the patch, keep the previous known-good main.cpp, and let the
            # next retry try again with an explicit note about what went wrong.
            logs.append("[fix_test_error] Implementation fix REJECTED: the model's "
                        "corrected code had no main() function -- keeping the previous "
                        "main.cpp unchanged instead of writing a broken build.")
            result["test_extra_info"] = (
                (extra_info or "")
                + "\nYour previous attempt returned implementation_code with no main() "
                  "function -- main.cpp must remain a complete, independently "
                  "compilable program. Include the full file with main() intact."
            )
        else:
            src_dir = state["src_dir"]
            _write(src_dir, "main.cpp", fixed_cpp)
            cpp_test_impl = remove_main_function(fixed_cpp)
            _write(src_dir, "main_test.cpp", cpp_test_impl)
            result["cpp_full"] = fixed_cpp
            result["cpp_test_impl"] = cpp_test_impl
        logs.append("[fix_test_error] Patched main.cpp -- it will be recompiled before "
                     "the next test run.")
    else:
        logs.append(f"[fix_test_error] {'HITL-guided' if extra_info else 'Auto'} retry: "
                     f"regenerated test.cpp.")

    result["logs"] = logs
    result["status"] = "test_fix_attempted"
    if not extra_info:
        result["test_retry_count"] = state.get("test_retry_count", 0) + 1
    return result


def human_review_test(state: MigrationState) -> dict:
    decision = interrupt({
        "stage": "test",
        "message": "The generated tests keep failing to compile/run after automatic "
                    "retries. Provide extra guidance for one more automated attempt, or "
                    "edit test.cpp directly.",
        "test_code": state["test_code"],
        "test_result": state["test_result"],
    })
    action = decision.get("action", "retry_with_info")
    result: dict[str, Any] = {
        "test_decision": action,
        "logs": [f"[human_review_test] User chose: {action}"],
    }
    if action == "retry_with_info":
        result["test_extra_info"] = decision.get("extra_info", "")
    else:
        result["test_edited_code"] = decision.get("edited_code", state["test_code"])
    return result


def route_after_test_hitl(state: MigrationState) -> str:
    return "extra_retry" if state.get("test_decision") == "retry_with_info" else "edited"


def apply_test_edit(state: MigrationState) -> dict:
    edited = state.get("test_edited_code", state.get("test_code", ""))
    _write(state["test_dir"], "test.cpp", edited)
    return {
        "test_code": edited,
        "logs": ["[apply_test_edit] Using human-edited test.cpp; recompiling once more."],
        "status": "test_edit_applied",
    }


def skip_test_execution(state: MigrationState) -> dict:
    reasons = []
    if not state["tool_status"]["g++"]:
        reasons.append("g++ is not installed")
    if not state["tool_status"]["gtest"]:
        reasons.append("GoogleTest is not installed")
    if not state.get("compile_result", {}).get("success", False):
        reasons.append("main.cpp did not compile successfully, so linking tests against it "
                        "would not be meaningful")
    return {
        "test_result": {"stage": "skipped", "success": False, "skipped": True, "reasons": reasons},
        "coverage_report": {"available": False, "note": "Test execution skipped: " + "; ".join(reasons)},
        "logs": [f"[skip_test_execution] Skipped: {reasons}"],
        "status": "test_execution_skipped",
    }


# ==========================================================================
# Final: assemble the human-readable report
# ==========================================================================

def finalize_report(state: MigrationState) -> dict:
    from agent.report import build_final_report  # local import avoids a cycle at module load
    report = build_final_report(state)
    reports_dir = state["reports_dir"]
    _write(reports_dir, "REPORT.md", report)
    return {
        "final_report": report,
        "logs": [f"[finalize_report] Wrote {reports_dir}/REPORT.md."],
        "status": "complete",
    }
