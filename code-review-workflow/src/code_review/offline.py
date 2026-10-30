"""Rule-based stand-in for Claude so the whole chain runs offline (tests, demos, CI).
It is a plumbing check, NOT a proxy for model quality: measure real quality with a live key."""
from __future__ import annotations

import json
import re
from typing import Any

from agentic_core import LLMResponse, MockClient, ToolCall

# (regex, category, severity, title, explanation, fix_description, patch)
RULES = [
    (r"\beval\(|\bexec\(", "security", "critical", "Dynamic code execution",
     "eval/exec on dynamic input allows arbitrary code execution.",
     "Replace with a safe parser such as ast.literal_eval or an explicit dispatch table.",
     "result = ast.literal_eval(expr)"),
    (r"(?i)(password|secret|api[_-]?key|token)\w*\s*=\s*[\"']\[REDACTED_SECRET\]", "security", "critical",
     "Hard-coded credential", "A credential is committed in source code and must be treated as leaked.",
     "Load the secret from the environment or a secret manager and rotate the exposed value.",
     'PASSWORD = os.environ["APP_PASSWORD"]'),
    (r"execute\(\s*f[\"']|execute\(.*[\"']\s*\+|execute\(.*%\s", "security", "critical", "SQL injection",
     "Query is built by string interpolation from variables.",
     "Use parameterised queries.", 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))'),
    (r"shell\s*=\s*True", "security", "high", "Shell injection risk",
     "shell=True with dynamic input lets an attacker inject commands.",
     "Pass an argument list and avoid the shell.", "subprocess.run(args, check=True)"),
    (r"yaml\.load\(|pickle\.loads?\(", "security", "high", "Unsafe deserialization",
     "Deserializing untrusted data can execute arbitrary code.",
     "Use yaml.safe_load or a data-only format such as JSON.", "data = yaml.safe_load(stream)"),
    (r"hashlib\.(md5|sha1)\(", "security", "medium", "Weak hash algorithm",
     "MD5/SHA-1 are unsuitable for passwords or integrity checks.",
     "Use a password hash (bcrypt/argon2) or SHA-256 for integrity.", "digest = hashlib.sha256(data).hexdigest()"),
    (r"def \w+\(.*=\s*(\[\]|\{\})", "bug", "high", "Mutable default argument",
     "The default object is shared across calls, causing state leaks between invocations.",
     "Default to None and create the object inside the function.", "def f(items=None):\n    items = items if items is not None else []"),
    (r"[=!]=\s*None\b", "bug", "low", "Comparison to None with ==",
     "Use identity comparison for None.", "Use `is None` / `is not None`.", "if value is None:"),
    (r"^\s*except\s*:", "reliability", "medium", "Bare except",
     "Catches SystemExit and KeyboardInterrupt and hides real errors.",
     "Catch specific exceptions and log them.", "except ValueError as exc:\n    logger.exception(exc)"),
    (r"except Exception\s*:\s*(pass)?\s*$", "reliability", "medium", "Silently swallowed exception",
     "Errors are ignored, making failures invisible.", "Log the exception or re-raise.",
     "except Exception:\n    logger.exception('operation failed')\n    raise"),
]
_LINE = re.compile(r"^\s*(\d+)\| ?(.*)$")


def _analyze(text: str) -> LLMResponse:
    issues = []
    for raw in text.splitlines():
        m = _LINE.match(raw)
        if not m:
            continue
        no, code = int(m[1]), m[2]
        if code.lstrip().startswith("#"):
            continue
        for pat, cat, sev, title, expl, *_ in RULES:
            if re.search(pat, code):
                issues.append({"line": no, "category": cat, "severity": sev, "title": title,
                               "explanation": expl, "evidence": code.strip(), "confidence": 0.85})
                break
    return LLMResponse(tool_calls=[ToolCall("report_issues", {"issues": issues})])


def _handler(req: dict[str, Any]) -> LLMResponse:
    tool, text = req.get("force_tool"), str(req["messages"][-1]["content"])
    if tool == "report_issues":
        return _analyze(text)
    if tool == "verify_issues":
        ids = [c["id"] for c in json.loads(re.search(r"<candidates>\n(.*)\n</candidates>", text, re.S)[1])]
        return LLMResponse(tool_calls=[ToolCall(tool, {"verdicts": [
            {"id": i, "valid": True, "reason": "confirmed by rule engine"} for i in ids]})])
    if tool == "suggest_fixes":
        conf = json.loads(re.search(r"<confirmed_issues>\n(.*)\n</confirmed_issues>", text, re.S)[1])
        by_title = {r[3]: r for r in RULES}
        return LLMResponse(tool_calls=[ToolCall(tool, {"fixes": [
            {"id": c["id"], "fix_description": by_title[c["title"]][5], "patch": by_title[c["title"]][6]}
            for c in conf if c["title"] in by_title]})])
    n = text.count('"severity"')
    return LLMResponse(text=f"The file has {n} confirmed issue(s). Address the highest-severity item first.")


def offline_client() -> MockClient:
    return MockClient(_handler)
