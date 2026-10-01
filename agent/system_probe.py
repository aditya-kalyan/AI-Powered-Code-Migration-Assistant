"""Local-machine probing: what compiler/testing/static-analysis tools exist,
plus small subprocess and source-code utilities shared across graph nodes.

Nothing in this module talks to an LLM -- it's pure "what does this machine
actually have installed" logic, kept separate so it's easy to unit test.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any

from agent.config import IS_WINDOWS, SUBPROCESS_TIMEOUT_SECONDS


# --------------------------------------------------------------------------
# Subprocess helpers
# --------------------------------------------------------------------------

def _resolve_relative_executable(cmd: list[str], cwd: str) -> list[str]:
    """Python's subprocess resolves a relative executable path (like './main'
    or '.\\main.exe') against the *parent* process's working directory, not
    against the `cwd=` kwarg passed to subprocess.run -- it explicitly does
    not consider `cwd` when searching for the executable. That means a
    freshly compiled binary sitting in a temp working directory is invoked
    with a path that's relative to wherever the Python process itself
    happens to be running, which is usually wrong and surfaces as
    "[WinError 2] The system cannot find the file specified" on Windows (or
    a bare FileNotFoundError on POSIX). Rewriting './x' / '.\\x' to an
    absolute path under `cwd` before invoking fixes that for every caller.
    """
    if not cmd:
        return cmd
    exe = cmd[0]
    if exe.startswith("./") or exe.startswith(".\\"):
        exe_name = exe[2:]
        return [os.path.join(cwd, exe_name)] + cmd[1:]
    return cmd


def run_cmd(cmd: list[str], cwd: str, timeout: int = SUBPROCESS_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Runs a command, always returning a structured result instead of raising,
    so graph nodes can branch on `.success` without try/except everywhere."""
    cmd = _resolve_relative_executable(cmd, cwd)
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except FileNotFoundError as exc:
        return {"command": cmd, "returncode": -1, "stdout": "", "stderr": str(exc), "success": False}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": f"Timed out after {timeout}s",
            "success": False,
        }


# --------------------------------------------------------------------------
# Tool detection
# --------------------------------------------------------------------------

def has_gpp() -> bool:
    return shutil.which("g++") is not None


def has_cppcheck() -> bool:
    return shutil.which("cppcheck") is not None


def has_gtest() -> bool:
    """GoogleTest has no single canonical marker (it's a source/header library,
    not always a pkg-config entry), so this checks a few signals in order of
    cheapness before falling back to the most reliable-but-expensive one: an
    actual trial compile+link.
    """
    # 1. pkg-config, if gtest was installed via a package manager that registers it
    if shutil.which("pkg-config"):
        try:
            out = subprocess.run(["pkg-config", "--exists", "gtest"], capture_output=True, timeout=5)
            if out.returncode == 0:
                return True
        except Exception:
            pass

    # 2. common header locations
    common_header_paths = [
        "/usr/include/gtest/gtest.h",
        "/usr/local/include/gtest/gtest.h",
        "/opt/homebrew/include/gtest/gtest.h",
    ]
    import os
    if any(os.path.exists(p) for p in common_header_paths):
        # Header presence alone doesn't guarantee the static libs are linkable,
        # but it's a strong signal; combine with a trial link if g++ exists.
        pass

    # 3. trial compile+link (authoritative, but needs g++)
    if has_gpp():
        with tempfile.TemporaryDirectory() as td:
            src = f"{td}/gtest_probe.cpp"
            with open(src, "w", encoding="utf-8") as f:
                f.write(
                    "#include <gtest/gtest.h>\n"
                    "TEST(Probe, Works) { EXPECT_TRUE(true); }\n"
                    "int main(int argc, char **argv) {\n"
                    "  ::testing::InitGoogleTest(&argc, argv);\n"
                    "  return RUN_ALL_TESTS();\n"
                    "}\n"
                )
            result = run_cmd(
                ["g++", src, "-o", f"{td}/gtest_probe", "-lgtest", "-lgtest_main", "-pthread"],
                cwd=td,
                timeout=20,
            )
            return result["success"]

    return any(os.path.exists(p) for p in common_header_paths)


def probe_tools() -> dict[str, bool]:
    return {"g++": has_gpp(), "gtest": has_gtest(), "cppcheck": has_cppcheck()}


def classify_tool_mode(tool_status: dict[str, bool]) -> str:
    """"none": nothing usable is present -> halt the pipeline entirely.
    "full": all three present -> fully automated pipeline.
    "partial": a mix -> generate everything an LLM can generate, but skip
      whichever *execution* steps depend on the missing tool(s)."""
    present = sum(tool_status.values())
    if present == 0:
        return "none"
    if present == len(tool_status):
        return "full"
    return "partial"


def install_steps_for(missing_tools: list[str]) -> list[str]:
    """OS-specific, human-readable installation steps for each missing tool."""
    steps: list[str] = []
    system = platform.system()

    if "g++" in missing_tools:
        if system == "Windows":
            steps.append(
                "Install a C++ compiler: get MSYS2 (https://www.msys2.org/) then run "
                "`pacman -S mingw-w64-ucrt-x86_64-gcc` and add its bin/ folder to PATH."
            )
        elif system == "Darwin":
            steps.append("Install Xcode command line tools: `xcode-select --install`.")
        else:
            steps.append("Install g++: `sudo apt install build-essential` (Debian/Ubuntu) "
                          "or the equivalent for your distro.")

    if "gtest" in missing_tools:
        if system == "Windows":
            steps.append(
                "Install GoogleTest via vcpkg: `vcpkg install gtest` then integrate it "
                "(`vcpkg integrate install`), or build it from source "
                "(https://github.com/google/googletest)."
            )
        elif system == "Darwin":
            steps.append("Install GoogleTest: `brew install googletest`.")
        else:
            steps.append("Install GoogleTest: `sudo apt install libgtest-dev` "
                          "(on some distros you then need to build the libs: see "
                          "https://github.com/google/googletest for the two-line cmake build).")

    if "cppcheck" in missing_tools:
        if system == "Windows":
            steps.append("Install cppcheck: download the installer from "
                          "https://cppcheck.sourceforge.io/ or `choco install cppcheck`.")
        elif system == "Darwin":
            steps.append("Install cppcheck: `brew install cppcheck`.")
        else:
            steps.append("Install cppcheck: `sudo apt install cppcheck`.")

    return steps


def collect_system_info() -> str:
    """Small JSON blob describing the host, handed to the LLM as porting context
    (e.g. so it can pick sane optimization flags)."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
    }
    for compiler, probe in (("g++", ["g++", "--version"]), ("clang++", ["clang++", "--version"])):
        path = shutil.which(compiler)
        if path:
            try:
                out = subprocess.run(probe, capture_output=True, text=True, timeout=5)
                first_line = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "unknown"
            except Exception:
                first_line = "present, version check failed"
            info[compiler] = {"path": path, "version": first_line}
    return json.dumps(info, indent=2)


# --------------------------------------------------------------------------
# Source manipulation
# --------------------------------------------------------------------------

_MAIN_FUNCTION_PATTERN = r"\bint\s+main\s*\([^)]*\)\s*\{"


def has_main_function(cpp_code: str) -> bool:
    """True if `cpp_code` defines a top-level int main(...). Used to validate
    LLM-produced "corrected implementation" text before it overwrites main.cpp
    -- an implementation fix that silently drops main() compiles into nothing
    linkable as a standalone program (MinGW/ld surfaces this as an opaque
    "undefined reference to WinMain", since it falls back to assuming a GUI
    entry point was intended once no main() is found)."""
    return re.search(_MAIN_FUNCTION_PATTERN, cpp_code) is not None


def remove_main_function(cpp_code: str) -> str:
    """Strips a top-level `int main(...) { ... }` block via brace counting, so
    the implementation can be #include-d into a test binary that supplies its
    own main() (gtest_main)."""
    match = re.search(_MAIN_FUNCTION_PATTERN, cpp_code)
    if not match:
        return cpp_code
    start = match.start()
    brace_start = cpp_code.find("{", match.start())
    depth = 0
    end = brace_start
    while end < len(cpp_code):
        if cpp_code[end] == "{":
            depth += 1
        elif cpp_code[end] == "}":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    return cpp_code[:start] + cpp_code[end:]


def _exe_path(workdir: str, name: str) -> str:
    """Absolute path to a built executable. Using an absolute path (rather
    than a relative `./name` or `.\\name`) sidesteps a real Windows
    `subprocess` gotcha: without `shell=True`, CreateProcess does not reliably
    resolve a relative executable path against `cwd`, which manifests as
    `[WinError 2] The system cannot find the file specified` even though the
    file is sitting right there. An absolute path removes the ambiguity on
    every platform."""
    exe = name + (".exe" if IS_WINDOWS else "")
    return os.path.join(workdir, exe)


def default_compile_command(workdir: str, source: str, output: str) -> list[str]:
    return ["g++", "-O3", "-std=c++17", source, "-o", _exe_path(workdir, output)]


def default_run_command(workdir: str, output: str) -> list[str]:
    return [_exe_path(workdir, output)]


def default_test_compile_command(workdir: str) -> list[str]:
    return ["g++", "-g", "-O0", "--coverage", "-std=c++17", "test.cpp", "-o",
            _exe_path(workdir, "test"), "-lgtest", "-lgtest_main", "-pthread"]


def default_test_run_command(workdir: str) -> list[str]:
    return [_exe_path(workdir, "test")]


def default_coverage_text_command(test_dir_rel: str) -> list[str]:
    """Plain-text summary (for parsing line/branch % into the report).

    This command is run with `cwd` set to the *project root* (the parent of
    both `src/` and `tests/`), not `tests/`. gcovr's `-r` root isn't just
    "where to look for .gcda/.gcno files" -- it's also the scope used to
    decide which source files are even eligible to appear in the report.
    The actual implementation lives in `src/main_test.cpp`, included into
    `test.cpp` via `../src/main_test.cpp` -- if the root were `tests/`, that
    file falls *outside* the root and gcovr silently drops it entirely,
    producing an all-zero report ("All coverage data is filtered out")
    even though the test binary was correctly instrumented and run.
    Rooting at the project root keeps both directories in scope.

    `test_dir_rel` is the tests/ folder's path relative to the project root
    (normally just "tests") -- passed as a positional search path so gcovr
    knows where the compiled .gcda/.gcno files actually are.

    The --exclude value is a regex, matched unanchored against the
    (now root-relative) file path, e.g. "tests/test.cpp". Anchoring it
    avoids the unrelated bug where a naive "test.cpp" pattern also matches
    "main_test.cpp" as a literal substring.
    """
    exclude_pattern = rf"(^|/){re.escape(test_dir_rel)}/test\.cpp$"
    return ["gcovr", "-r", ".", "--branches", "--exclude", exclude_pattern, test_dir_rel]


def default_coverage_html_command(test_dir_rel: str, output_html_path: str) -> list[str]:
    """Same rooting rationale as `default_coverage_text_command`.
    `output_html_path` should be relative to the project root (the `cwd`
    this command runs in), typically `tests/reports/index.html`."""
    exclude_pattern = rf"(^|/){re.escape(test_dir_rel)}/test\.cpp$"
    return ["gcovr", "-r", ".", "--branches", "--exclude", exclude_pattern,
            "--html", "--html-details", "-o", output_html_path, test_dir_rel]


def parse_gcovr_summary(text: str) -> dict[str, Any]:
    """Best-effort extraction of overall line/branch coverage percentages from
    gcovr's default text summary. Returns {} if the format isn't recognized
    (still fine -- the raw text is kept in the report regardless)."""
    result: dict[str, Any] = {}
    line_match = re.search(r"^lines:\s*([\d.]+)%", text, re.IGNORECASE | re.MULTILINE)
    branch_match = re.search(r"^branches:\s*([\d.]+)%", text, re.IGNORECASE | re.MULTILINE)
    if line_match:
        result["line_coverage_pct"] = float(line_match.group(1))
    if branch_match:
        result["branch_coverage_pct"] = float(branch_match.group(1))
    return result
