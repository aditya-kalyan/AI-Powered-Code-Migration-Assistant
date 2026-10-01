"""The single state object threaded through every node in the LangGraph graph.

`total=False` because most fields are populated incrementally as the pipeline
progresses -- a node early in the graph legitimately hasn't set fields that a
later node will fill in.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class MigrationState(TypedDict, total=False):
    # ---- inputs ----
    python_code: str
    model: str
    workdir: str
    max_retries: int
    judge_threshold: int

    # ---- tool detection ----
    tool_status: dict[str, bool]          # {"g++": bool, "gtest": bool, "cppcheck": bool}
    tool_mode: str                        # "none" | "partial" | "full"
    install_steps: list[str]
    system_info: str

    # ---- output layout (all absolute paths, created by check_tools) ----
    src_dir: str                          # <workdir>/src      -- main.cpp, main_test.cpp
    test_dir: str                         # <workdir>/tests    -- test.cpp
    reports_dir: str                      # <workdir>/reports  -- REPORT.md, coverage, cppcheck

    # ---- porting ----
    cpp_full: str                         # main.cpp content (with main())
    cpp_test_impl: str                    # main.cpp with main() stripped, for test linking

    # ---- judge gate (semantic relevance / RAII / memory mgmt) ----
    judge_result: dict[str, Any]
    judge_retry_count: int
    judge_decision: str                   # "retry_with_info" | "edit"  (set by HITL resume)
    judge_extra_info: str
    judge_edited_code: str

    # ---- compile gate ----
    compile_result: dict[str, Any]
    compile_command: list[str]
    run_command: list[str]
    compile_retry_count: int
    compile_decision: str
    compile_extra_info: str
    compile_edited_code: str

    # ---- static analysis (cppcheck) ----
    cppcheck_result: dict[str, Any]

    # ---- test generation ----
    test_code: str
    test_coverage_justification: str
    test_compile_command: list[str]
    test_run_command: list[str]
    coverage_command: list[str]
    coverage_html_command: list[str]

    # ---- test compile/run gate ----
    test_result: dict[str, Any]
    coverage_report: dict[str, Any]
    test_retry_count: int
    test_decision: str
    test_extra_info: str
    test_edited_code: str

    # ---- bookkeeping ----
    logs: Annotated[list[str], operator.add]
    status: str
    final_report: str
