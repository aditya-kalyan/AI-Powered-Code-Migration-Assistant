"""Gradio front-end for the migration agent.

Design note on the HITL panel: all three gated stages (judge / compile /
test) pause the graph via the *same* `interrupt()` payload shape
(`{"stage", "message", ...code/result...}`) and resume via the *same*
`Command(resume={"action", "extra_info", "edited_code"})` shape. That lets one
generic panel + one generic resume handler serve all three stages instead of
tripling the UI code.

Design note on output location: every run writes into `src/`, `tests/`, and
`reports/` subfolders of the *current working directory* (wherever `python
app.py` was launched from) -- not a system temp directory. Re-running
overwrites the previous run's files in those same three folders. See
`agent/nodes.py::check_tools` for where the folders are created.
"""

from __future__ import annotations

import os
import uuid

import gradio as gr
from langgraph.types import Command

from agent.config import DEFAULT_MODEL, JUDGE_PASS_THRESHOLD, MAX_AUTO_RETRIES
from agent.graph import build_graph

GRAPH = build_graph()

# The directory the user launched `python app.py` from -- this is where the
# src/, tests/, and reports/ subfolders will be created.
PROJECT_ROOT = os.getcwd()

SAMPLE_PYTHON = '''class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None


def reverse_dll(head):
    current = head
    temp = None
    while current:
        temp = current.prev
        current.prev = current.next
        current.next = temp
        current = current.prev
    if temp:
        head = temp.prev
    return head


def build_list(values):
    head = None
    tail = None
    for v in values:
        node = Node(v)
        if head is None:
            head = node
            tail = node
        else:
            tail.next = node
            node.prev = tail
            tail = node
    return head


def to_list(head):
    out = []
    while head:
        out.append(head.value)
        head = head.next
    return out


if __name__ == "__main__":
    head = build_list([1, 2, 3, 4, 5])
    head = reverse_dll(head)
    print(to_list(head))
'''

# Number of output components every _render() call must produce, in this
# exact order. Keeping this as one source of truth avoids the two sides
# (the tuple _render returns, and the `outputs=[...]` list in build_ui)
# silently drifting out of sync.
_OUTPUT_ORDER = (
    "logs", "cpp_code", "compile_text", "cppcheck_text", "test_code", "test_text",
    "coverage_text", "status_md", "report_md", "paths_md", "hitl_visible",
    "hitl_message", "hitl_code",
)


def _fmt(d: dict | None, keys: tuple[str, ...] = ("stage", "success", "returncode", "stdout", "stderr")) -> str:
    if not d:
        return "(none)"
    lines = []
    for k in keys:
        if k in d and d[k]:
            v = d[k]
            lines.append(f"--- {k} ---\n{v}" if k in ("stdout", "stderr") else f"{k}: {v}")
    return "\n".join(lines) if lines else "(no output)"


def _paths_markdown(result: dict) -> str:
    src_dir = result.get("src_dir")
    test_dir = result.get("test_dir")
    reports_dir = result.get("reports_dir")
    if not src_dir:
        return ""
    return (
        f"**Output folders for this run** (under `{result.get('workdir', PROJECT_ROOT)}`):\n"
        f"- `src/` -> {src_dir}\n"
        f"- `tests/` -> {test_dir}\n"
        f"- `reports/` -> {reports_dir}"
    )


def _empty_result(message: str) -> dict:
    """Standard shape for the 13 output values when nothing has run yet
    (validation errors, missing thread_id, etc.) -- keeps every early-return
    branch below in sync with `_OUTPUT_ORDER` automatically."""
    return dict(
        logs=message, cpp_code="", compile_text="", cppcheck_text="", test_code="",
        test_text="", coverage_text="", status_md=message, report_md="", paths_md="",
        hitl_visible=gr.update(visible=False), hitl_message="", hitl_code="",
    )


def _as_tuple(values: dict) -> tuple:
    return tuple(values[k] for k in _OUTPUT_ORDER)


def _render(result: dict) -> tuple:
    """Maps a graph result dict (from either the initial invoke or a resume)
    to every output component's new value, in `_OUTPUT_ORDER`. Kept as one
    function so the start and resume handlers stay in sync."""
    logs = "\n".join(result.get("logs", []))
    cpp_code = result.get("cpp_full", "")
    compile_text = _fmt(result.get("compile_result"))
    cppcheck = result.get("cppcheck_result") or {}
    cppcheck_text = cppcheck.get("raw", "(not run)") if cppcheck.get("ran") else "(skipped -- cppcheck not installed)"
    test_code = result.get("test_code", "")
    test_text = _fmt(result.get("test_result"))
    coverage = result.get("coverage_report") or {}
    coverage_text = (
        f"line: {coverage.get('line_coverage_pct', 'n/a')}%  branch: {coverage.get('branch_coverage_pct', 'n/a')}%"
        + (f"  (html: {coverage['html_report_path']})" if coverage.get("html_generated") else "")
        if coverage.get("available") else coverage.get("note", "(not available)")
    )
    paths_md = _paths_markdown(result)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        stage = payload["stage"]
        message = payload["message"]
        code_for_review = payload.get("cpp_code") or payload.get("test_code") or ""
        extra_ctx = ""
        if stage == "judge":
            jr = payload.get("judge_result", {})
            extra_ctx = f"Judge scores -- overall: {jr.get('overall_score')}, feedback: {jr.get('feedback')}"
        elif stage == "compile":
            extra_ctx = _fmt(payload.get("compile_result"))
        elif stage == "test":
            extra_ctx = _fmt(payload.get("test_result"))

        values = dict(
            logs=logs, cpp_code=cpp_code, compile_text=compile_text, cppcheck_text=cppcheck_text,
            test_code=test_code, test_text=test_text, coverage_text=coverage_text,
            status_md=f"⏸️ Paused for human review at the **{stage}** stage.",
            report_md="", paths_md=paths_md,
            hitl_visible=gr.update(visible=True),
            hitl_message=f"### Human review needed: `{stage}` stage\n\n{message}\n\n```\n{extra_ctx}\n```",
            hitl_code=code_for_review,
        )
        return _as_tuple(values)

    if result.get("tool_mode") == "none":
        steps = "\n".join(f"- {s}" for s in result.get("install_steps", []))
        values = dict(
            logs=logs, cpp_code="", compile_text="", cppcheck_text="", test_code="", test_text="",
            coverage_text="",
            status_md=f"### No required tools found\n\nInstall at least one of the following before running:\n\n{steps}",
            report_md="", paths_md=paths_md,
            hitl_visible=gr.update(visible=False), hitl_message="", hitl_code="",
        )
        return _as_tuple(values)

    values = dict(
        logs=logs, cpp_code=cpp_code, compile_text=compile_text, cppcheck_text=cppcheck_text,
        test_code=test_code, test_text=test_text, coverage_text=coverage_text,
        status_md=f"Pipeline status: **{result.get('status', 'unknown')}**",
        report_md=result.get("final_report", ""), paths_md=paths_md,
        hitl_visible=gr.update(visible=False), hitl_message="", hitl_code="",
    )
    return _as_tuple(values)


def start_pipeline(python_code: str, model: str, max_retries: int, judge_threshold: int):
    if not python_code.strip():
        return _as_tuple(_empty_result("Paste some Python code first.")) + ("",)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    init_state = {
        "python_code": python_code,
        "model": model or DEFAULT_MODEL,
        "workdir": PROJECT_ROOT,
        "max_retries": int(max_retries),
        "judge_threshold": int(judge_threshold),
        "logs": [],
    }

    try:
        result = GRAPH.invoke(init_state, config)
    except Exception as exc:  # surfaced to the UI rather than crashing the server
        return _as_tuple(_empty_result(f"Pipeline error: {exc}")) + (thread_id,)

    return _render(result) + (thread_id,)


def resume_pipeline(action: str, extra_info: str, edited_code: str, thread_id: str):
    if not thread_id:
        return _as_tuple(_empty_result("No active run to resume.")) + (thread_id,)

    config = {"configurable": {"thread_id": thread_id}}
    resume_payload = {"action": action, "extra_info": extra_info, "edited_code": edited_code}
    try:
        result = GRAPH.invoke(Command(resume=resume_payload), config)
    except Exception as exc:
        return _as_tuple(_empty_result(f"Resume error: {exc}")) + (thread_id,)

    return _render(result) + (thread_id,)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Python -> C++ Migration Agent") as demo:
        gr.Markdown(
            "# Python -> C++ Code Migration Agent\n"
            "LangGraph pipeline: tool detection -> LLM port -> **LLM-as-judge gate** "
            "(semantic relevance / RAII / memory management) -> compile & run "
            "(auto-retry) -> cppcheck static analysis -> GoogleTest generation -> "
            "compile & run tests + coverage (auto-retry). Every gated stage escalates "
            "to a human-in-the-loop checkpoint after repeated automatic failures. "
            f"Output files are written to `src/`, `tests/`, and `reports/` under "
            f"`{PROJECT_ROOT}`. See README.md for the full architecture diagram."
        )

        thread_id_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=1):
                python_in = gr.Code(label="Python source", language="python", value=SAMPLE_PYTHON, lines=18)
                model_in = gr.Textbox(label="OpenAI model", value=DEFAULT_MODEL)
                retries_in = gr.Slider(label="Max auto-fix retries per stage", minimum=0, maximum=5,
                                        step=1, value=MAX_AUTO_RETRIES)
                threshold_in = gr.Slider(label="Judge pass threshold", minimum=0, maximum=100,
                                          step=5, value=JUDGE_PASS_THRESHOLD)
                run_btn = gr.Button("Run pipeline", variant="primary")
                paths_out = gr.Markdown()

            with gr.Column(scale=1):
                cpp_out = gr.Code(label="Generated main.cpp", language="cpp", lines=18)
                compile_out = gr.Textbox(label="Compile & run result", lines=6)
                cppcheck_out = gr.Textbox(label="cppcheck (static analysis, informational)", lines=4)

        status_out = gr.Markdown()

        with gr.Group(visible=False) as hitl_group:
            hitl_message_out = gr.Markdown()
            hitl_code_box = gr.Code(label="Code under review (editable)", language="cpp", lines=16)
            extra_info_box = gr.Textbox(
                label="Additional guidance for one more automated attempt",
                placeholder="e.g. 'the off-by-one is in the boundary check, not the loop init'",
                lines=2,
            )
            with gr.Row():
                retry_btn = gr.Button("Retry once more with this guidance", variant="primary")
                edit_btn = gr.Button("Use my edited code above instead")

        with gr.Row():
            test_out = gr.Code(label="Generated test.cpp", language="cpp", lines=16)
            with gr.Column():
                test_result_out = gr.Textbox(label="Test compile & run result", lines=6)
                coverage_out = gr.Textbox(label="Coverage", lines=2)

        with gr.Accordion("Pipeline log", open=False):
            logs_out = gr.Textbox(label="", lines=12, show_label=False)

        report_out = gr.Markdown(label="Final report")

        # Order MUST match _OUTPUT_ORDER exactly.
        run_outputs = [
            logs_out, cpp_out, compile_out, cppcheck_out, test_out, test_result_out,
            coverage_out, status_out, report_out, paths_out, hitl_group, hitl_message_out,
            hitl_code_box,
        ]

        run_btn.click(
            start_pipeline,
            inputs=[python_in, model_in, retries_in, threshold_in],
            outputs=run_outputs + [thread_id_state],
        )

        retry_btn.click(
            lambda info, code, tid: resume_pipeline("retry_with_info", info, code, tid),
            inputs=[extra_info_box, hitl_code_box, thread_id_state],
            outputs=run_outputs + [thread_id_state],
        )
        edit_btn.click(
            lambda info, code, tid: resume_pipeline("edit", info, code, tid),
            inputs=[extra_info_box, hitl_code_box, thread_id_state],
            outputs=run_outputs + [thread_id_state],
        )

    return demo


demo = build_ui()
