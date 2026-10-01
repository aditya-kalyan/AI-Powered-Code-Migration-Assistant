"""Builds the final human-readable Markdown report from the finished graph state."""

from __future__ import annotations

from agent.state import MigrationState


def _bullet(label: str, ok: bool) -> str:
    return f"- {'✅' if ok else '❌'} {label}"


def build_final_report(state: MigrationState) -> str:
    lines: list[str] = ["# Code Migration Report", ""]

    # --- Output locations (answers "where are my files?") ---
    lines += ["## Output locations", ""]
    lines += [
        f"- Ported C++ (`main.cpp`, `main_test.cpp`): `{state.get('src_dir', '?')}`",
        f"- Generated tests (`test.cpp`): `{state.get('test_dir', '?')}`",
        f"- Reports (`REPORT.md`, `index.html`, `cppcheck.txt`): `{state.get('reports_dir', '?')}`",
        "",
    ]

    # --- Tools ---
    ts = state.get("tool_status", {})
    lines += ["## Tool availability", ""]
    lines += [_bullet(f"`{name}`", present) for name, present in ts.items()]
    lines += ["", f"Mode: **{state.get('tool_mode', 'unknown')}**", ""]
    if state.get("install_steps"):
        lines += ["### Install missing tools", ""]
        lines += [f"1. {s}" for s in state["install_steps"]]
        lines.append("")

    # --- Judge ---
    judge = state.get("judge_result")
    if judge:
        lines += ["## Code quality judge", ""]
        lines += [
            f"- Overall score: **{judge.get('overall_score')}/100**",
            f"- Semantic relevance: {judge.get('semantic_relevance')}/100",
            f"- RAII compliance: {judge.get('raii_score')}/100",
            f"- Memory management: {judge.get('memory_management_score')}/100",
        ]
        if judge.get("feedback"):
            lines.append(f"- Feedback: {judge['feedback']}")
        if state.get("judge_decision"):
            lines.append(f"- Human-in-the-loop was invoked; user chose: `{state['judge_decision']}`")
        lines.append("")

    # --- Compile ---
    compile_result = state.get("compile_result")
    lines += ["## C++ compilation", ""]
    if compile_result is None:
        lines.append("Not attempted (g++ unavailable).")
    else:
        success = compile_result.get("success", False)
        lines.append(_bullet("Compiled and ran successfully", success))
        if not success:
            lines.append(f"- Failure stage: `{compile_result.get('stage')}`")
            lines.append(f"- Return code: {compile_result.get('returncode')}")
            if compile_result.get("stderr"):
                lines.append(f"- stderr:\n```\n{compile_result['stderr'][:2000]}\n```")
        if state.get("compile_decision"):
            lines.append(f"- Human-in-the-loop was invoked; user chose: `{state['compile_decision']}`")
    lines.append("")

    # --- cppcheck ---
    cppcheck = state.get("cppcheck_result", {})
    lines += ["## Static analysis (cppcheck)", ""]
    if not cppcheck.get("ran", False):
        lines.append("Skipped (cppcheck unavailable).")
    else:
        lines.append(f"- Errors: {cppcheck.get('error_count', 0)}")
        lines.append(f"- Warnings: {cppcheck.get('warning_count', 0)}")
        if cppcheck.get("raw"):
            lines.append(f"- Raw output:\n```\n{cppcheck['raw'][:2000]}\n```")
    lines.append("")

    # --- Tests ---
    lines += ["## Generated tests", ""]
    if state.get("test_coverage_justification"):
        lines.append(f"LLM coverage justification: {state['test_coverage_justification']}")
    test_result = state.get("test_result")
    if test_result is None:
        lines.append("Test code was generated but never compiled (see tool availability above).")
    elif test_result.get("skipped"):
        lines.append("Test execution skipped: " + "; ".join(test_result.get("reasons", [])))
        lines.append("")
        lines.append("### Manual steps once the missing tool(s) are installed")
        lines.append(f"From `{state.get('test_dir', '<tests folder>')}`, run:")
        lines.append("```bash")
        if state.get("test_compile_command"):
            lines.append(" ".join(state["test_compile_command"]))
        if state.get("test_run_command"):
            lines.append(" ".join(state["test_run_command"]))
        if state.get("coverage_command"):
            lines.append(" ".join(state["coverage_command"]))
        lines.append("```")
    else:
        success = test_result.get("success", False)
        lines.append(_bullet("Tests compiled and passed", success))
        if not success:
            lines.append(f"- Failure stage: `{test_result.get('stage')}`")
            if test_result.get("stderr"):
                lines.append(f"- stderr:\n```\n{test_result['stderr'][:2000]}\n```")
            lines.append("- Only the last generated `test.cpp` is kept; it may still contain "
                          "failing cases -- see stderr above.")
        if state.get("test_decision"):
            lines.append(f"- Human-in-the-loop was invoked; user chose: `{state['test_decision']}`")

        coverage = state.get("coverage_report", {})
        if coverage.get("available"):
            lines.append("")
            lines.append("### Coverage")
            if "line_coverage_pct" in coverage:
                lines.append(f"- Line coverage: {coverage['line_coverage_pct']}%")
            if "branch_coverage_pct" in coverage:
                lines.append(f"- Branch coverage: {coverage['branch_coverage_pct']}%")
            if coverage.get("html_generated"):
                lines.append(f"- HTML report: `{coverage.get('html_report_path')}`")
        elif coverage.get("note"):
            lines.append(f"- Coverage note: {coverage['note']}")

    lines.append("")
    lines.append(f"Final status: **{state.get('status', 'unknown')}**")
    return "\n".join(lines)
