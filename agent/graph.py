"""Assembles the LangGraph `StateGraph`. See `agent/nodes.py`'s module
docstring for the naming convention behind the auto-retry -> HITL -> "_final"
/ "_extra" node families.

Full topology (see README.md for the rendered diagram):

    START
      -> check_tools --[halt]--> END
                      --[proceed]--> port_to_cpp
      port_to_cpp -> judge_cpp
      judge_cpp --[pass]--> compile_gate
                --[retry]--> port_to_cpp                       (loop, <=3x)
                --[hitl]--> human_review_judge (INTERRUPT)
      human_review_judge --[extra_retry]--> port_to_cpp_extra --> judge_cpp_final --> compile_gate
                         --[edited]-->      apply_judge_edit  -----------------------> compile_gate

      compile_gate --[compile]--> compile_and_run_cpp
                   --[skip]--> cppcheck_gate
      compile_and_run_cpp --[success]--> cppcheck_gate
                          --[retry]--> fix_cpp_error --> compile_and_run_cpp   (loop, <=3x)
                          --[hitl]--> human_review_compile (INTERRUPT)
      human_review_compile --[extra_retry]--> fix_cpp_error_extra --> compile_cpp_final --> cppcheck_gate
                           --[edited]-->      apply_compile_edit  -------------------------> cppcheck_gate

      cppcheck_gate --[run]--> cppcheck_analysis --> generate_tests
                    --[skip]--> skip_cppcheck    --> generate_tests

      generate_tests -> test_gate
      test_gate --[compile]--> compile_and_run_tests
                --[skip]--> skip_test_execution --> finalize_report
      compile_and_run_tests --[success]--> finalize_report
                            --[retry]--> fix_test_error --> compile_and_run_tests   (loop, <=3x)
                            --[hitl]--> human_review_test (INTERRUPT)
      human_review_test --[extra_retry]--> fix_test_error_extra --> compile_and_run_tests_final --> finalize_report
                        --[edited]-->      apply_test_edit  ------------------------------------> finalize_report

      finalize_report -> END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent import nodes as N
from agent.state import MigrationState


def build_graph():
    g = StateGraph(MigrationState)

    # -- tool detection --
    g.add_node("check_tools", N.check_tools)

    # -- judge gate --
    g.add_node("port_to_cpp", N.port_to_cpp)
    g.add_node("judge_cpp", N.judge_cpp)
    g.add_node("judge_cpp_final", N.judge_cpp)          # reused fn, unconditional edge out
    g.add_node("human_review_judge", N.human_review_judge)
    g.add_node("port_to_cpp_extra", N.port_to_cpp_extra)
    g.add_node("apply_judge_edit", N.apply_judge_edit)

    # -- compile gate --
    g.add_node("compile_gate", N.compile_gate)
    g.add_node("compile_and_run_cpp", N.compile_and_run_cpp)
    g.add_node("compile_cpp_final", N.compile_and_run_cpp)  # reused fn, unconditional edge out
    g.add_node("fix_cpp_error", N.fix_cpp_error)
    g.add_node("fix_cpp_error_extra", N.fix_cpp_error)      # reused fn, extra_info branches inside
    g.add_node("human_review_compile", N.human_review_compile)
    g.add_node("apply_compile_edit", N.apply_compile_edit)

    # -- static analysis --
    g.add_node("cppcheck_analysis", N.cppcheck_analysis)
    g.add_node("skip_cppcheck", N.skip_cppcheck)

    # -- test generation + gate --
    g.add_node("generate_tests", N.generate_tests)
    g.add_node("compile_and_run_tests", N.compile_and_run_tests)
    g.add_node("compile_and_run_tests_final", N.compile_and_run_tests)  # reused fn
    g.add_node("fix_test_error", N.fix_test_error)
    g.add_node("fix_test_error_extra", N.fix_test_error)                # reused fn
    g.add_node("human_review_test", N.human_review_test)
    g.add_node("apply_test_edit", N.apply_test_edit)
    g.add_node("skip_test_execution", N.skip_test_execution)

    # -- report --
    g.add_node("finalize_report", N.finalize_report)

    # ---------------------------------------------------------------- edges

    g.add_edge(START, "check_tools")
    g.add_conditional_edges("check_tools", N.route_after_tool_check, {"halt": END, "proceed": "port_to_cpp"})

    g.add_edge("port_to_cpp", "judge_cpp")
    g.add_conditional_edges(
        "judge_cpp", N.route_after_judge,
        {"pass": "compile_gate", "retry": "port_to_cpp", "hitl": "human_review_judge"},
    )
    g.add_conditional_edges(
        "human_review_judge", N.route_after_judge_hitl,
        {"extra_retry": "port_to_cpp_extra", "edited": "apply_judge_edit"},
    )
    g.add_edge("port_to_cpp_extra", "judge_cpp_final")
    g.add_edge("judge_cpp_final", "compile_gate")
    g.add_edge("apply_judge_edit", "compile_gate")

    g.add_conditional_edges(
        "compile_gate", N.route_after_compile_gate,
        {"compile": "compile_and_run_cpp", "skip": "cppcheck_gate_router"},
    )
    g.add_conditional_edges(
        "compile_and_run_cpp", N.route_after_compile,
        {"success": "cppcheck_gate_router", "retry": "fix_cpp_error", "hitl": "human_review_compile"},
    )
    g.add_edge("fix_cpp_error", "compile_and_run_cpp")
    g.add_conditional_edges(
        "human_review_compile", N.route_after_compile_hitl,
        {"extra_retry": "fix_cpp_error_extra", "edited": "apply_compile_edit"},
    )
    g.add_edge("fix_cpp_error_extra", "compile_cpp_final")
    g.add_edge("compile_cpp_final", "cppcheck_gate_router")
    g.add_edge("apply_compile_edit", "compile_cpp_final")

    # cppcheck_gate is a pure routing function (no state mutation needed), so
    # it's wired as a passthrough node feeding a conditional edge.
    g.add_node("cppcheck_gate_router", lambda state: {})
    g.add_conditional_edges(
        "cppcheck_gate_router", N.cppcheck_gate,
        {"run": "cppcheck_analysis", "skip": "skip_cppcheck"},
    )
    g.add_edge("cppcheck_analysis", "generate_tests")
    g.add_edge("skip_cppcheck", "generate_tests")

    g.add_node("test_gate_router", lambda state: {})
    g.add_edge("generate_tests", "test_gate_router")
    g.add_conditional_edges(
        "test_gate_router", N.test_gate,
        {"compile": "compile_and_run_tests", "skip": "skip_test_execution"},
    )
    g.add_conditional_edges(
        "compile_and_run_tests", N.route_after_test_compile,
        {"success": "finalize_report", "retry": "fix_test_error", "hitl": "human_review_test"},
    )
    g.add_edge("fix_test_error", "compile_and_run_tests")
    g.add_conditional_edges(
        "human_review_test", N.route_after_test_hitl,
        {"extra_retry": "fix_test_error_extra", "edited": "apply_test_edit"},
    )
    g.add_edge("fix_test_error_extra", "compile_and_run_tests_final")
    g.add_edge("compile_and_run_tests_final", "finalize_report")
    g.add_edge("apply_test_edit", "compile_and_run_tests_final")
    g.add_edge("skip_test_execution", "finalize_report")

    g.add_edge("finalize_report", END)

    return g.compile(checkpointer=MemorySaver())
