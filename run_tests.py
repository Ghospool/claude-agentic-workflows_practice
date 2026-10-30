"""Run every project's offline test suite. Works on Windows, macOS and Linux; needs no API key.

    python run_tests.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECTS = ["customer-support-router", "code-review-workflow", "data-processing-pipeline",
            "research-assistant-agent"]


def main() -> int:
    failed = []
    for name in PROJECTS:
        proj = ROOT / name
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "agentic-core" / "src"), str(proj / "src")]),
                   AUDIT_LOG_PATH=os.devnull)
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=proj, env=env)
        if r.returncode:
            failed.append(name)
    print("\nFAILED: " + ", ".join(failed) if failed else "\nAll suites passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
