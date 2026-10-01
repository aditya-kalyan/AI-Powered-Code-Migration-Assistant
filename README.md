# Python → C++ Migration Agent

A portfolio-ready LangGraph project that turns Python code into C++, validates the result with an LLM judge, compiles it, generates a GoogleTest suite, and reports quality metrics in a local Gradio UI.

## Why this project stands out

- End-to-end AI-assisted software migration workflow
- Human-in-the-loop safety checks at critical stages
- Automatic compile/test repair loops with clear failure reporting
- Local artifact generation for C++ source, tests, and coverage reports
- Clean single-app interface for demonstrations and portfolio showcases

## Tech stack

- Python 3.10+
- LangGraph
- OpenAI API
- Gradio
- g++ / GoogleTest / cppcheck / gcovr (optional, for full automation)

## Quick start

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Add your OpenAI key

Create a `.env` file or set the environment variable before launching:

```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

Or:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

A sample template is included in `.env.example`.

### 4) Run the app

```bash
python app.py
```

The app opens the browser automatically at:

```text
http://127.0.0.1:7860
```

## How it works

1. Paste Python code into the UI.
2. The system checks local tool availability.
3. An LLM converts the Python into C++.
4. An LLM-as-judge scores the result for code quality.
5. The generated C++ is compiled and repaired if needed.
6. cppcheck and test generation run if the required tools are available.
7. Final reports and generated source files are saved locally.

## Repository layout

```text
app.py
agent/
  config.py
  graph.py
  llm_client.py
  nodes.py
  prompts.py
  report.py
  state.py
  system_probe.py
ui/
  blocks.py
src/
  main.cpp
  main_test.cpp
tests/
  test.cpp
  reports/
    REPORT.md
    cppcheck.txt
    index.html
README.md
requirements.txt
.env.example
LICENSE
```

## Output folders

Every run writes generated artifacts under the directory where you started the app:

```text
<project-root>/
src/
tests/
reports/
```

This is intentional and makes the project easy to inspect after each run.

## Optional tooling for full automation

For the strongest end-to-end demo, install:

- g++
- GoogleTest
- cppcheck
- gcovr

Without these, the app still generates code and reports, but some execution steps are skipped gracefully.

## Portfolio notes

This repo is good for a GitHub portfolio because it demonstrates:

- AI integration with real software engineering workflows
- safety checks and validation loops
- multi-stage orchestration with LangGraph
- practical use of compiler/test tools in a local app

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
