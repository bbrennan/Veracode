# “””
veracode_lite.py

A pre-flight scanner that mimics Veracode Static Analysis findings for
Python + SnowSQL codebases, with a confidence score for how likely each
finding is a real issue vs. noise/false positive.

Drop in repo root. Zero deps for core scan (pandas optional for DataFrame).

CLI:
python veracode_lite.py .
python veracode_lite.py . –json findings.json –min-severity 3
python veracode_lite.py . –likely-real-only

Jupyter:
from veracode_lite import Scanner
s = Scanner(”.”).run()
df = s.to_dataframe()
df[(df.confidence > 0.7) & (df.severity >= 4)]
s.summary()
s.explain(df.iloc[0])    # show signals for a finding
“””
from **future** import annotations

import ast
import json
import os
import re
import sys
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Iterable
from collections import Counter, defaultdict

# —————————————————————————

# Finding model + confidence helpers

# —————————————————————————

CONF_REAL = 0.70
CONF_REVIEW = 0.30

def _confidence_label(c: float) -> str:
if c >= CONF_REAL:
return “likely_real”
if c >= CONF_REVIEW:
return “needs_review”
return “likely_false_positive”

def _clamp(c: float) -> float:
return max(0.05, min(0.99, c))

# —————————————————————————

# Human-readable CWE descriptions (the “what is this and why does it matter”)

# —————————————————————————

CWE_INFO: dict[str, dict[str, str]] = {
“CWE-22”: {
“description”: (
“Path Traversal. The application uses user-controllable input to “
“construct a filesystem path without restricting the result to an “
“allowed base directory. Sequences like ‘../’ or absolute paths can “
“escape the intended scope.”
),
“impact”: (
“An attacker can read or write arbitrary files the application’s “
“process has access to: source code, config files, secrets on disk, “
“or overwriting application binaries.”
),
},
“CWE-78”: {
“description”: (
“OS Command Injection. User-controllable input is passed to a shell “
“interpreter (subprocess shell=True, os.system, os.popen) where “
“metacharacters like ;, &&, | can chain additional commands.”
),
“impact”: (
“Full arbitrary command execution on the host with the application’s “
“privileges. Typically the highest-impact web application weakness “
“after authentication bypass.”
),
},
“CWE-89”: {
“description”: (
“SQL Injection. User-controllable input is concatenated or “
“interpolated into a SQL statement instead of being bound as a “
“parameter. The database driver cannot distinguish data from query “
“structure.”
),
“impact”: (
“Authentication bypass, exfiltration of every row in the database, “
“modification or destruction of data, and on some database engines “
“remote code execution via stored procedures or file writes.”
),
},
“CWE-94”: {
“description”: (
“Code Injection. The application turns attacker-controlled data into “
“executable code via functions like eval(), exec(), or compile().”
),
“impact”: (
“Arbitrary code execution within the application process — same “
“privileges as the running service. Effectively equivalent to giving “
“the attacker a Python shell on the box.”
),
},
“CWE-295”: {
“description”: (
“Improper Certificate Validation. TLS certificate verification is “
“explicitly disabled (verify=False) or misconfigured, so the client “
“accepts any certificate the server presents.”
),
“impact”: (
“An attacker on the network path (compromised Wi-Fi, DNS hijack, “
“malicious proxy) can intercept and modify traffic by presenting a “
“forged certificate. Credentials and tokens in the connection are “
“exposed.”
),
},
“CWE-327”: {
“description”: (
“Use of Broken or Risky Cryptographic Algorithm. An algorithm with “
“known weaknesses (MD5, SHA1, DES, RC4) is used in a context where “
“security properties like collision resistance or confidentiality “
“are required.”
),
“impact”: (
“Depending on the use: password hashes can be cracked at scale, “
“signatures can be forged via collision, or encrypted data can be “
“recovered. Often a downgrade vector even if not directly broken.”
),
},
“CWE-330”: {
“description”: (
“Use of Insufficiently Random Values. A non-cryptographic PRNG “
“(Python’s random module) is used to generate values that must be “
“unpredictable: session tokens, password reset links, CSRF nonces, “
“API keys.”
),
“impact”: (
“Given a few observed outputs an attacker can predict future values “
“and forge sessions, hijack reset flows, or guess API keys. The “
“secrets module is the correct choice for any of these uses.”
),
},
“CWE-489”: {
“description”: (
“Active Debug Code. Debugging artifacts left in production source: “
“breakpoint(), pdb.set_trace(), Flask debug=True, exposed debug “
“routes, or verbose error pages.”
),
“impact”: (
“Ranges from information disclosure (stack traces revealing paths “
“and library versions) to interactive code execution (Werkzeug debug “
“console gives a shell to anyone who can reach it).”
),
},
“CWE-502”: {
“description”: (
“Deserialization of Untrusted Data. The application deserializes “
“data through a format that supports arbitrary object construction “
“(pickle, dill, yaml.load with FullLoader) without verifying the “
“source.”
),
“impact”: (
“Code execution on deserialization, triggered by crafted bytes in “
“the input stream. Often invisible because no obvious eval/exec is “
“in the source. Pickle is essentially equivalent to running code “
“from the source of the serialized data.”
),
},
“CWE-611”: {
“description”: (
“XML External Entity (XXE). An XML parser processes external entity “
“references in untrusted documents. Stdlib parsers (xml.etree, “
“lxml without defaults set) are vulnerable; defusedxml is the “
“hardened drop-in replacement.”
),
“impact”: (
“Local file disclosure (entity references like file:///etc/passwd), “
“server-side request forgery, and denial of service via the “
“billion-laughs entity expansion attack.”
),
},
“CWE-732”: {
“description”: (
“Incorrect Permission Assignment for Critical Resource. A file, “
“directory, IPC resource, or database object is granted permissions “
“broader than required for its function.”
),
“impact”: (
“Other local users or processes can read sensitive data, modify “
“configuration, or replace executables. In Snowflake/database “
“contexts: GRANT TO PUBLIC exposes data across the entire account.”
),
},
“CWE-798”: {
“description”: (
“Use of Hard-coded Credentials. Authentication material — passwords, “
“API keys, signing keys, database credentials — is embedded as a “
“string literal in source code rather than read from environment, “
“secrets manager, or key-pair file.”
),
“impact”: (
“Anyone with read access to the source (current and former employees, “
“anyone who pulls the repo, anyone who finds the artifact in a “
“container layer or build output) holds production credentials. “
“Rotation requires a code change.”
),
},
“CWE-863”: {
“description”: (
“Incorrect Authorization (used here as a quality check for “
“destructive SQL without a WHERE clause). Strictly a safety/quality “
“issue rather than an authorization weakness, but Veracode-adjacent “
“tooling commonly flags unconditional DELETE/TRUNCATE.”
),
“impact”: (
“Catastrophic data loss when run against the wrong environment or “
“table. Not exploitable from outside, but a frequent cause of “
“self-inflicted production incidents.”
),
},
}

@dataclass
class Finding:
cwe: str
name: str
severity: int           # 1..5 (Veracode V1=Info, V5=Very High)
file: str
line: int
snippet: str
message: str
confidence: float       # 0..1
confidence_label: str
signals: list = field(default_factory=list)
suggested_fix: str = “”
mitigation_template: Optional[str] = None
description: str = “”   # populated from CWE_INFO in **post_init**
impact: str = “”

```
def __post_init__(self):
    info = CWE_INFO.get(self.cwe, {})
    if not self.description:
        self.description = info.get("description", "")
    if not self.impact:
        self.impact = info.get("impact", "")

def to_dict(self):
    return asdict(self)
```

# —————————————————————————

# Heuristic helpers

# —————————————————————————

USER_INPUT_HINTS = {
“request”, “args”, “form”, “params”, “input”, “stdin”, “argv”, “env”,
“user”, “payload”, “body”, “query”, “header”, “cookie”, “post”, “get”,
“json_body”, “raw”, “external”,
}

CONFIG_HINTS = {“config”, “settings”, “constants”, “default”, “static”, “internal”}

CRED_NAMES = re.compile(
r”(?:^|*)(password|passwd|pwd|secret|api[*-]?key|access[*-]?key|”
r”auth[*-]?token|private[*-]?key|client[*-]?secret|credential|sf[*-]?password|”
r”snowflake[*-]?password)(?:_|$)”,
re.IGNORECASE,
)

EXAMPLE_VALUE = re.compile(
r”(EXAMPLE|PLACEHOLDER|YOUR[*-]|XXX|TODO|FAKE|DUMMY|CHANGE[*-]?ME|<.*>)”,
re.IGNORECASE,
)
AWS_TEST_KEY = “AKIAIOSFODNN7EXAMPLE”

TEST_PATH = re.compile(r”(^|[/\])(tests?[/\]|test_|conftest.py|_test.py$)”, re.IGNORECASE)
MIGRATION_PATH = re.compile(r”(migrations?|alembic)[/\]”, re.IGNORECASE)

def _name_hints_user_input(s: str) -> bool:
s = s.lower()
return any(h in s for h in USER_INPUT_HINTS)

def _name_hints_config(s: str) -> bool:
s = s.lower()
return any(h in s for h in CONFIG_HINTS)

def _is_test_file(path: str) -> bool:
return bool(TEST_PATH.search(path))

def _is_migration_file(path: str) -> bool:
return bool(MIGRATION_PATH.search(path))

def _walk_names(node) -> list[str]:
“”“Collect identifier-like strings inside an expression.”””
out = []
for sub in ast.walk(node):
if isinstance(sub, ast.Name):
out.append(sub.id)
elif isinstance(sub, ast.Attribute):
out.append(sub.attr)
return out

def _get_full_attr(node) -> str:
“”“Return dotted name for an Attribute/Name node, or ‘’ if not resolvable.”””
parts = []
cur = node
while isinstance(cur, ast.Attribute):
parts.append(cur.attr)
cur = cur.value
if isinstance(cur, ast.Name):
parts.append(cur.id)
return “.”.join(reversed(parts))
return “”

def _kw(call: ast.Call, name: str) -> Optional[ast.expr]:
for k in call.keywords:
if k.arg == name:
return k.value
return None

def *is_const(node, type*=None) -> bool:
if not isinstance(node, ast.Constant):
return False
return type_ is None or isinstance(node.value, type_)

def _is_dynamic_string(node) -> bool:
“”“f-string, %-format, .format(), or string concatenation.”””
if isinstance(node, ast.JoinedStr):
# f-string — dynamic only if any part is non-constant
return any(not isinstance(v, ast.Constant) for v in node.values)
if isinstance(node, ast.BinOp):
# ’foo ’ + var  or  ‘foo %s’ % var
return True
if isinstance(node, ast.Call):
f = node.func
if isinstance(f, ast.Attribute) and f.attr == “format”:
return True
return False

# —————————————————————————

# Python AST scanner

# —————————————————————————

class _PyVisitor(ast.NodeVisitor):
def **init**(self, source: str, file: str):
self.lines = source.splitlines()
self.file = file
self.is_test = _is_test_file(file)
self.is_migration = _is_migration_file(file)
self.findings: list[Finding] = []
# track whether defusedxml / safe alternatives are imported
self.imports_safe_yaml = False
self.imports_defusedxml = False

```
def snippet(self, line: int) -> str:
    if 1 <= line <= len(self.lines):
        return self.lines[line - 1].strip()[:240]
    return ""

def add(self, **kw):
    kw["confidence"] = _clamp(kw["confidence"])
    kw["confidence_label"] = _confidence_label(kw["confidence"])
    kw.setdefault("file", self.file)
    kw.setdefault("snippet", self.snippet(kw["line"]))
    self.findings.append(Finding(**kw))

# ---- import tracking ------------------------------------------------
def visit_ImportFrom(self, node):
    if node.module == "defusedxml" or (node.module or "").startswith("defusedxml."):
        self.imports_defusedxml = True
    self.generic_visit(node)

def visit_Import(self, node):
    for alias in node.names:
        if alias.name.startswith("defusedxml"):
            self.imports_defusedxml = True
    self.generic_visit(node)

# ---- main dispatch --------------------------------------------------
def visit_Call(self, node):
    self._check_sql_execute(node)
    self._check_subprocess(node)
    self._check_eval_exec(node)
    self._check_deserialize(node)
    self._check_hashlib(node)
    self._check_requests_verify(node)
    self._check_chmod(node)
    self._check_xml_parse(node)
    self._check_open_traversal(node)
    self._check_debug_calls(node)
    self._check_random_security(node)
    self._check_snowflake_connect(node)
    self.generic_visit(node)

def visit_Assign(self, node):
    self._check_hardcoded_cred(node)
    self.generic_visit(node)

# =====================================================================
# Rule: CWE-89 SQL Injection on cursor.execute / executemany
# =====================================================================
def _check_sql_execute(self, call: ast.Call):
    f = call.func
    if not isinstance(f, ast.Attribute):
        return
    if f.attr not in ("execute", "executemany", "execute_string"):
        return
    if not call.args:
        return
    sql_arg = call.args[0]
    if not _is_dynamic_string(sql_arg):
        return

    signals = []
    conf = 0.55  # base for "execute() with dynamic string"
    signals.append(("base: dynamic SQL passed to .execute()", +0.55))

    # Positive: tainted-looking variable inside the string
    names = _walk_names(sql_arg)
    if any(_name_hints_user_input(n) for n in names):
        conf += 0.20
        signals.append(("variable name suggests user input", +0.20))

    # Positive: it's an f-string with formatted parts (vs. just BinOp constants)
    if isinstance(sql_arg, ast.JoinedStr) and any(isinstance(v, ast.FormattedValue) for v in sql_arg.values):
        conf += 0.05
        signals.append(("f-string with interpolated values", +0.05))

    # Negative: bound params later in same call (psycopg2 style)
    if len(call.args) >= 2 and isinstance(call.args[1], (ast.Tuple, ast.List, ast.Dict, ast.Name)):
        conf -= 0.25
        signals.append(("execute() also passes bind params (likely safe)", -0.25))

    # Negative: SQLAlchemy text() with :param markers
    src = ast.unparse(sql_arg) if hasattr(ast, "unparse") else ""
    if "text(" in src and re.search(r":\w+", src):
        conf -= 0.40
        signals.append(("SQLAlchemy text() with :bound params", -0.40))

    # Negative: all parts literal
    if isinstance(sql_arg, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in sql_arg.values):
        conf -= 0.30
        signals.append(("all string parts are literal constants", -0.30))

    # Negative: file context
    if self.is_test:
        conf -= 0.20
        signals.append(("in test file", -0.20))
    if self.is_migration:
        conf -= 0.15
        signals.append(("in migration file", -0.15))

    self.add(
        cwe="CWE-89",
        name="SQL Injection",
        severity=5,
        line=call.lineno,
        message="Dynamic SQL passed to .execute()/.executemany().",
        confidence=conf,
        signals=signals,
        suggested_fix=(
            "Use parameterized queries:\n"
            "  cursor.execute('SELECT * FROM t WHERE id = %s', (user_id,))\n"
            "Or SQLAlchemy: session.execute(text('... :id ...'), {'id': user_id})"
        ),
        mitigation_template=_MIT_CWE89,
    )

# =====================================================================
# Rule: CWE-78 Command Injection via subprocess shell=True / os.system
# =====================================================================
def _check_subprocess(self, call: ast.Call):
    full = _get_full_attr(call.func)
    is_subprocess = full.startswith("subprocess.") and full.split(".")[-1] in {
        "run", "call", "check_call", "check_output", "Popen"
    }
    is_os_system = full in ("os.system", "os.popen")

    if not (is_subprocess or is_os_system):
        return

    # Determine danger
    cmd_arg = call.args[0] if call.args else None
    shell_kw = _kw(call, "shell")
    shell_true = isinstance(shell_kw, ast.Constant) and shell_kw.value is True

    # os.system / os.popen are inherently shell-true
    if is_os_system:
        shell_true = True

    if not shell_true and is_subprocess:
        # subprocess with list arg and shell=False is safe
        return

    signals = []
    conf = 0.55
    signals.append(("base: shell-style command execution", +0.55))

    if cmd_arg is not None:
        if _is_dynamic_string(cmd_arg):
            conf += 0.20
            signals.append(("command is dynamically built (concat/f-string)", +0.20))
            names = _walk_names(cmd_arg)
            if any(_name_hints_user_input(n) for n in names):
                conf += 0.15
                signals.append(("variable name suggests user input", +0.15))
            else:
                if any(_name_hints_config(n) for n in names):
                    conf -= 0.20
                    signals.append(("variables look config-derived", -0.20))
        elif _is_const(cmd_arg, str):
            conf -= 0.30
            signals.append(("command is a literal constant string", -0.30))

    if self.is_test:
        conf -= 0.20
        signals.append(("in test file", -0.20))

    self.add(
        cwe="CWE-78",
        name="OS Command Injection",
        severity=5,
        line=call.lineno,
        message=f"{full} called with shell semantics.",
        confidence=conf,
        signals=signals,
        suggested_fix=(
            "Avoid shell=True. Pass a list of args:\n"
            "  subprocess.run(['ping', '-c', '1', host], shell=False, check=True)"
        ),
    )

# =====================================================================
# Rule: CWE-94 Code Injection via eval / exec
# =====================================================================
def _check_eval_exec(self, call: ast.Call):
    if not isinstance(call.func, ast.Name):
        return
    if call.func.id not in ("eval", "exec", "compile"):
        return
    if not call.args:
        return

    arg = call.args[0]
    signals = []
    conf = 0.65
    signals.append((f"base: {call.func.id}() invoked", +0.65))

    # ast.literal_eval is in different rule; here it's true eval/exec
    if _is_const(arg, str):
        conf -= 0.30
        signals.append(("argument is a literal string constant", -0.30))
    else:
        names = _walk_names(arg)
        if any(_name_hints_user_input(n) for n in names):
            conf += 0.20
            signals.append(("argument variable suggests user input", +0.20))

    if self.is_test:
        conf -= 0.20
        signals.append(("in test file", -0.20))

    self.add(
        cwe="CWE-94",
        name="Code Injection (eval/exec)",
        severity=5,
        line=call.lineno,
        message=f"Use of {call.func.id}() can execute arbitrary code.",
        confidence=conf,
        signals=signals,
        suggested_fix=(
            "Use ast.literal_eval() for literal data structures, or a real expression\n"
            "parser (simpleeval, asteval) with a whitelist of allowed names."
        ),
    )

# =====================================================================
# Rule: CWE-502 Unsafe Deserialization
# =====================================================================
def _check_deserialize(self, call: ast.Call):
    full = _get_full_attr(call.func)

    # pickle.loads / pickle.load / cPickle / dill
    if full in ("pickle.loads", "pickle.load", "cPickle.loads", "cPickle.load",
                "dill.loads", "dill.load", "_pickle.loads", "_pickle.load"):
        self._add_pickle_finding(call, full, kind="pickle")
        return

    # yaml.load without SafeLoader
    if full in ("yaml.load",):
        loader_kw = _kw(call, "Loader")
        loader_name = ""
        if loader_kw is not None:
            loader_name = _get_full_attr(loader_kw) or (loader_kw.id if isinstance(loader_kw, ast.Name) else "")
        if "Safe" in loader_name:
            return  # SafeLoader/CSafeLoader: actually safe
        self._add_pickle_finding(call, full, kind="yaml")
        return

    # joblib.load / torch.load — pickle under the hood, lower confidence
    if full in ("joblib.load", "torch.load"):
        self._add_pickle_finding(call, full, kind="ml-load")
        return

def _add_pickle_finding(self, call, full, kind):
    signals = []
    conf = 0.55
    signals.append((f"base: {full}() deserializes pickle/yaml", +0.55))

    # If first arg is an open() of a *literal* file path, lower confidence
    if call.args:
        a0 = call.args[0]
        if isinstance(a0, ast.Call) and _get_full_attr(a0.func) == "open":
            if a0.args and _is_const(a0.args[0], str):
                conf -= 0.10
                signals.append(("loaded from literal file path", -0.10))

        # Variable named like request/payload — strongly suggests untrusted
        names = _walk_names(a0)
        if any(_name_hints_user_input(n) for n in names):
            conf += 0.25
            signals.append(("source variable suggests user input", +0.25))

    if kind == "ml-load":
        conf -= 0.15
        signals.append(("ML loader (joblib/torch) — typically trained artifacts", -0.15))

    if self.is_test:
        conf -= 0.20
        signals.append(("in test file", -0.20))

    self.add(
        cwe="CWE-502",
        name="Deserialization of Untrusted Data",
        severity=5,
        line=call.lineno,
        message=f"{full}() can execute arbitrary code on hostile input.",
        confidence=conf,
        signals=signals,
        suggested_fix=(
            "Prefer JSON for data interchange. For YAML use yaml.safe_load().\n"
            "For ML model loads, document provenance (signed bucket, checksum)."
        ),
        mitigation_template=_MIT_CWE502 if kind == "ml-load" else None,
    )

# =====================================================================
# Rule: CWE-327 Weak Crypto (md5, sha1)
# =====================================================================
def _check_hashlib(self, call: ast.Call):
    full = _get_full_attr(call.func)
    weak = None
    if full in ("hashlib.md5",):
        weak = "md5"
    elif full in ("hashlib.sha1",):
        weak = "sha1"
    elif full == "hashlib.new" and call.args and _is_const(call.args[0], str):
        v = call.args[0].value.lower()
        if v in ("md5", "sha1", "md4"):
            weak = v
    if not weak:
        return

    signals = []
    conf = 0.55
    signals.append((f"base: hashlib.{weak} flagged as weak", +0.55))

    # Negative: usedforsecurity=False
    ufs = _kw(call, "usedforsecurity")
    if isinstance(ufs, ast.Constant) and ufs.value is False:
        conf -= 0.50
        signals.append(("usedforsecurity=False (Python 3.9+)", -0.50))

    # Look at surrounding line for "cache", "etag", "dedup" hints
    snippet = self.snippet(call.lineno).lower()
    for hint in ("cache", "etag", "dedup", "fingerprint", "checksum"):
        if hint in snippet:
            conf -= 0.20
            signals.append((f"snippet mentions '{hint}' (non-security use)", -0.20))
            break

    # Positive: nearby words like password, token, signature
    for hint in ("password", "passwd", "token", "signature", "auth"):
        if hint in snippet:
            conf += 0.20
            signals.append((f"snippet mentions '{hint}' (security use)", +0.20))
            break

    if self.is_test:
        conf -= 0.20
        signals.append(("in test file", -0.20))

    self.add(
        cwe="CWE-327",
        name="Use of Broken/Risky Crypto",
        severity=4,
        line=call.lineno,
        message=f"Weak hash algorithm: {weak}",
        confidence=conf,
        signals=signals,
        suggested_fix=(
            "For passwords: use bcrypt or argon2.\n"
            "For non-security hashing on Python 3.9+, pass usedforsecurity=False."
        ),
        mitigation_template=_MIT_CWE327,
    )

# =====================================================================
# Rule: CWE-330 Insecure Random in security context
# =====================================================================
def _check_random_security(self, call: ast.Call):
    full = _get_full_attr(call.func)
    if not (full.startswith("random.") and full.split(".")[-1] in
            {"random", "randint", "choice", "choices", "sample", "uniform",
             "randrange", "getrandbits"}):
        return

    snippet = self.snippet(call.lineno).lower()
    sec_hints = ("token", "secret", "password", "passwd", "session",
                 "csrf", "nonce", "salt", "reset", "otp", "api_key")

    if not any(h in snippet for h in sec_hints):
        return  # don't flag generic random use

    signals = [("random.* used in apparent security context", +0.55)]
    conf = 0.55
    for h in sec_hints:
        if h in snippet:
            conf += 0.15
            signals.append((f"snippet mentions '{h}'", +0.15))
            break

    if self.is_test:
        conf -= 0.20
        signals.append(("in test file", -0.20))

    self.add(
        cwe="CWE-330",
        name="Insufficiently Random Values",
        severity=3,
        line=call.lineno,
        message="random module used for security-sensitive value.",
        confidence=conf,
        signals=signals,
        suggested_fix="Use the secrets module: secrets.token_urlsafe(32)",
    )

# =====================================================================
# Rule: CWE-295 Disabled cert verification
# =====================================================================
def _check_requests_verify(self, call: ast.Call):
    full = _get_full_attr(call.func)
    if not (full.startswith("requests.") or full.startswith("httpx.") or
            full.endswith(".session.get") or full.endswith(".session.post")):
        return
    v = _kw(call, "verify")
    if not (isinstance(v, ast.Constant) and v.value is False):
        return

    signals = [("verify=False disables TLS validation", +0.85)]
    conf = 0.85
    if self.is_test:
        conf -= 0.30
        signals.append(("in test file", -0.30))

    self.add(
        cwe="CWE-295",
        name="Improper Certificate Validation",
        severity=4,
        line=call.lineno,
        message=f"{full}(..., verify=False)",
        confidence=conf,
        signals=signals,
        suggested_fix="Remove verify=False, or pin a CA bundle: verify='/path/to/ca.crt'",
    )

# =====================================================================
# Rule: CWE-732 Permissive chmod
# =====================================================================
def _check_chmod(self, call: ast.Call):
    if _get_full_attr(call.func) != "os.chmod":
        return
    if len(call.args) < 2:
        return
    mode = call.args[1]
    if not isinstance(mode, ast.Constant) or not isinstance(mode.value, int):
        return
    m = mode.value
    # world-writable bits
    if not (m & 0o002 or m & 0o020):
        return

    signals = [(f"base: chmod mode {oct(m)} grants broad write", +0.80)]
    conf = 0.80
    if self.is_test:
        conf -= 0.30
        signals.append(("in test file", -0.30))

    self.add(
        cwe="CWE-732",
        name="Incorrect Permission Assignment",
        severity=4,
        line=call.lineno,
        message=f"Permissive file mode: {oct(m)}",
        confidence=conf,
        signals=signals,
        suggested_fix="Use restrictive modes like 0o640 or 0o600.",
    )

# =====================================================================
# Rule: CWE-611 XXE via stdlib XML parsers
# =====================================================================
def _check_xml_parse(self, call: ast.Call):
    full = _get_full_attr(call.func)
    unsafe_modules = ("xml.etree.ElementTree.", "xml.dom.minidom.", "xml.sax.", "lxml.etree.")
    unsafe_funcs = ("parse", "fromstring", "parseString", "XMLParser")
    if not any(full.startswith(m) for m in unsafe_modules):
        return
    if full.split(".")[-1] not in unsafe_funcs:
        return
    if self.imports_defusedxml:
        return  # likely using defused alternatives

    signals = [("XML parser used without defusedxml", +0.65)]
    conf = 0.65
    if self.is_test:
        conf -= 0.20
        signals.append(("in test file", -0.20))

    self.add(
        cwe="CWE-611",
        name="XML External Entity (XXE)",
        severity=4,
        line=call.lineno,
        message=f"{full} can be vulnerable to XXE / billion-laughs.",
        confidence=conf,
        signals=signals,
        suggested_fix=(
            "Use defusedxml:\n"
            "  import defusedxml.ElementTree as ET\n"
            "  ET.fromstring(data)"
        ),
    )

# =====================================================================
# Rule: CWE-22 Path Traversal via open() with dynamic path
# =====================================================================
def _check_open_traversal(self, call: ast.Call):
    if not (isinstance(call.func, ast.Name) and call.func.id == "open"):
        return
    if not call.args:
        return
    path_arg = call.args[0]
    if not _is_dynamic_string(path_arg):
        return

    signals = []
    conf = 0.45
    signals.append(("base: open() with dynamic path", +0.45))

    names = _walk_names(path_arg)
    if any(_name_hints_user_input(n) for n in names):
        conf += 0.30
        signals.append(("path variable suggests user input", +0.30))
    elif any(_name_hints_config(n) for n in names):
        conf -= 0.30
        signals.append(("path variables look config-derived", -0.30))

    if self.is_test:
        conf -= 0.20
        signals.append(("in test file", -0.20))

    self.add(
        cwe="CWE-22",
        name="Path Traversal",
        severity=5,
        line=call.lineno,
        message="open() called with dynamically constructed path.",
        confidence=conf,
        signals=signals,
        suggested_fix=(
            "Validate the resolved path stays within an allowed base directory:\n"
            "  target = os.path.realpath(os.path.join(base, name))\n"
            "  if not target.startswith(base + os.sep): abort(400)"
        ),
        mitigation_template=_MIT_CWE22,
    )

# =====================================================================
# Rule: CWE-489 Active debug code
# =====================================================================
def _check_debug_calls(self, call: ast.Call):
    full = _get_full_attr(call.func)
    debug = full in ("pdb.set_trace", "ipdb.set_trace", "pudb.set_trace")
    builtin_breakpoint = isinstance(call.func, ast.Name) and call.func.id == "breakpoint"
    if not (debug or builtin_breakpoint):
        return

    signals = [("debug call left in code", +0.85)]
    conf = 0.85
    if self.is_test:
        conf -= 0.50
        signals.append(("in test file (debug there is normal)", -0.50))

    self.add(
        cwe="CWE-489",
        name="Active Debug Code",
        severity=3,
        line=call.lineno,
        message=f"{full or 'breakpoint'}() left in source.",
        confidence=conf,
        signals=signals,
        suggested_fix="Remove the debug call before merging.",
    )

# =====================================================================
# Rule: CWE-798 Hardcoded credentials
# =====================================================================
def _check_hardcoded_cred(self, node: ast.Assign):
    if not _is_const(node.value, str):
        return
    value = node.value.value
    if not value or len(value) < 4:
        return

    for target in node.targets:
        name = _get_full_attr(target) or (target.id if isinstance(target, ast.Name) else "")
        if not name:
            continue
        if not CRED_NAMES.search(name):
            continue

        signals = []
        conf = 0.65
        signals.append((f"base: variable '{name}' looks like a credential", +0.65))

        if EXAMPLE_VALUE.search(value) or AWS_TEST_KEY in value:
            conf -= 0.50
            signals.append(("value looks like example/placeholder", -0.50))
        if value.startswith(("$", "{{", "%(")):
            conf -= 0.40
            signals.append(("value looks like template/env-var marker", -0.40))
        if self.is_test:
            conf -= 0.30
            signals.append(("in test file", -0.30))
        if len(value) >= 20 and re.search(r"[A-Za-z]", value) and re.search(r"\d", value):
            conf += 0.10
            signals.append(("value has password-like complexity", +0.10))

        self.add(
            cwe="CWE-798",
            name="Hardcoded Credentials",
            severity=5,
            line=node.lineno,
            message=f"Credential-like assignment: {name} = '...'",
            confidence=conf,
            signals=signals,
            suggested_fix=(
                "Read from environment or secrets manager:\n"
                f"  {name.split('.')[-1]} = os.environ['{name.split('.')[-1].upper()}']"
            ),
        )

# =====================================================================
# Rule: Snowflake connector with hardcoded password kwarg
# =====================================================================
def _check_snowflake_connect(self, call: ast.Call):
    full = _get_full_attr(call.func)
    if full not in ("snowflake.connector.connect", "connector.connect"):
        return
    pwd = _kw(call, "password")
    if pwd is None:
        return
    if not (isinstance(pwd, ast.Constant) and isinstance(pwd.value, str) and pwd.value):
        return

    signals = [("snowflake.connector.connect(password='...') hardcoded", +0.85)]
    conf = 0.85
    if EXAMPLE_VALUE.search(pwd.value):
        conf -= 0.50
        signals.append(("password looks like placeholder", -0.50))
    if self.is_test:
        conf -= 0.30
        signals.append(("in test file", -0.30))

    self.add(
        cwe="CWE-798",
        name="Hardcoded Snowflake Credential",
        severity=5,
        line=call.lineno,
        message="snowflake.connector.connect called with literal password.",
        confidence=conf,
        signals=signals,
        suggested_fix=(
            "Use key-pair auth or read password from env / secrets manager:\n"
            "  snowflake.connector.connect(user=..., password=os.environ['SF_PASSWORD'], ...)"
        ),
    )
```

# —————————————————————————

# SnowSQL / .sql file scanner (line-based regex)

# —————————————————————————

*SQL_F_INTERP = re.compile(r”{[A-Za-z*][\w.]*}”)  # {var} markers in templated SQL
_DESTRUCTIVE = re.compile(r”^\s*(DROP|TRUNCATE|DELETE\s+FROM)\s+”, re.IGNORECASE)
_HAS_WHERE = re.compile(r”\bWHERE\b”, re.IGNORECASE)
_GRANT_PUBLIC = re.compile(r”GRANT\s+.+\bTO\s+(ROLE\s+)?PUBLIC\b”, re.IGNORECASE)
_SQL_CRED = re.compile(r”\b(password|secret)\s*=\s*[’"][^’"]{4,}[’"]”, re.IGNORECASE)

def _scan_sql_file(path: str) -> list[Finding]:
out: list[Finding] = []
try:
text = Path(path).read_text(errors=“replace”)
except OSError:
return out
lines = text.splitlines()
is_test = _is_test_file(path)

```
for i, line in enumerate(lines, start=1):
    s = line.strip()
    if not s or s.startswith("--"):
        continue

    # Templated SQL with {var} markers
    if _SQL_F_INTERP.search(line):
        conf = 0.60 + (0.20 if "execute" not in line.lower() else 0.0)
        signals = [("Python-style {var} interpolation in SQL file", +0.60)]
        if is_test:
            conf -= 0.20
            signals.append(("in test file", -0.20))
        out.append(Finding(
            cwe="CWE-89",
            name="SQL Injection (templated SQL)",
            severity=5,
            file=path,
            line=i,
            snippet=s[:240],
            message="SQL contains {var}-style interpolation; ensure caller binds parameters.",
            confidence=_clamp(conf),
            confidence_label=_confidence_label(_clamp(conf)),
            signals=signals,
            suggested_fix="Use parameterized SnowSQL bindings, e.g. cursor.execute(sql, {'id': id_})",
            mitigation_template=_MIT_CWE89,
        ))

    # Destructive without WHERE (multi-line statements aren't perfectly handled)
    if _DESTRUCTIVE.search(s):
        # Look ahead a few lines for WHERE before next ;
        block = " ".join(lines[i-1:i-1+10])
        stmt = block.split(";", 1)[0]
        if not _HAS_WHERE.search(stmt) and "DROP" not in s.upper():
            conf = 0.55
            signals = [("destructive statement without WHERE", +0.55)]
            if is_test or _is_migration_file(path):
                conf -= 0.30
                signals.append(("in test/migration file", -0.30))
            out.append(Finding(
                cwe="CWE-863",
                name="Destructive SQL without WHERE",
                severity=3,
                file=path,
                line=i,
                snippet=s[:240],
                message="DELETE/TRUNCATE without a WHERE clause.",
                confidence=_clamp(conf),
                confidence_label=_confidence_label(_clamp(conf)),
                signals=signals,
                suggested_fix="Add a WHERE clause, or confirm full-table action is intentional.",
            ))

    # GRANT TO PUBLIC
    if _GRANT_PUBLIC.search(line):
        out.append(Finding(
            cwe="CWE-732",
            name="Overly Permissive GRANT",
            severity=4,
            file=path,
            line=i,
            snippet=s[:240],
            message="GRANT to PUBLIC role exposes privilege broadly.",
            confidence=_clamp(0.85 - (0.30 if is_test else 0.0)),
            confidence_label=_confidence_label(_clamp(0.85 - (0.30 if is_test else 0.0))),
            signals=[("GRANT ... TO PUBLIC", +0.85)],
            suggested_fix="Grant to a specific role instead of PUBLIC.",
        ))

    # Hardcoded credential in SQL
    if _SQL_CRED.search(line):
        out.append(Finding(
            cwe="CWE-798",
            name="Hardcoded Credential in SQL",
            severity=5,
            file=path,
            line=i,
            snippet=s[:240],
            message="Credential literal in SQL file.",
            confidence=_clamp(0.80 - (0.30 if is_test else 0.0)),
            confidence_label=_confidence_label(_clamp(0.80 - (0.30 if is_test else 0.0))),
            signals=[("password=... literal in SQL", +0.80)],
            suggested_fix="Use Snowflake's secret-manager or environment-substituted SnowSQL vars.",
        ))

return out
```

# —————————————————————————

# Mitigation templates

# —————————————————————————

_MIT_CWE89 = (
“By Design. The SQL at [file:line] uses parameterized binding via “
“[psycopg2 %s / SQLAlchemy text() :param / Snowflake connector qmark]. “
“User input is bound at the protocol layer; the f-string contains only “
“static SQL fragments and bound-parameter placeholders.”
)

_MIT_CWE327 = (
“By Design. The MD5/SHA1 hash at [file:line] is used as a [cache key / “
“ETag / dedup hash] with no security-relevant property. “
“Python 3.9+: usedforsecurity=False has been added to make this explicit.”
)

_MIT_CWE22 = (
“By Environment. The path at [file:line] is constructed from “
“application-controlled config (loaded from [env / SSM / secrets]). “
“No portion is influenced by HTTP request input or external sources.”
)

_MIT_CWE502 = (
“By Design. The deserialization at [file:line] loads a serialized “
“[scikit-learn / PyTorch] model retrieved from [S3 bucket] via IAM-restricted “
“role. Bucket has versioning + public-access-blocked + write restricted to “
“the training pipeline IAM role. Artifact integrity verified via “
“[SHA-256 checksum / signed manifest] before load.”
)

# —————————————————————————

# Scanner

# —————————————————————————

DEFAULT_EXCLUDE_DIRS = {
“.git”, “.venv”, “venv”, “env”, “**pycache**”, “node_modules”,
“.tox”, “.mypy_cache”, “.pytest_cache”, “build”, “dist”, “.eggs”,
“site-packages”, “.ipynb_checkpoints”, “.cache”, “.local”,
“vendor”, “third_party”, “.terraform”,
}

class Scanner:
def **init**(
self,
root: str | os.PathLike,
exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
include_tests: bool = True,
):
self.root = Path(root).resolve()
self.exclude_dirs = set(exclude_dirs)
self.include_tests = include_tests
self.findings: list[Finding] = []

```
# ---- file discovery -------------------------------------------------
def _iter_files(self):
    for dirpath, dirnames, filenames in os.walk(self.root):
        dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
        for fn in filenames:
            if fn.endswith((".py", ".sql")):
                full = os.path.join(dirpath, fn)
                if not self.include_tests and _is_test_file(full):
                    continue
                yield full

def _scan_python(self, path: str) -> list[Finding]:
    try:
        src = Path(path).read_text(errors="replace")
        tree = ast.parse(src, filename=path)
    except (SyntaxError, OSError, ValueError):
        return []
    rel = os.path.relpath(path, self.root)
    v = _PyVisitor(src, rel)
    # Some generated files (e.g. parser tables) have very deep nesting.
    # Bump the recursion limit briefly and skip on overflow.
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 5000))
    try:
        v.visit(tree)
    except RecursionError:
        return []
    finally:
        sys.setrecursionlimit(old_limit)
    return v.findings

def _scan_sql(self, path: str) -> list[Finding]:
    rel = os.path.relpath(path, self.root)
    out = _scan_sql_file(path)
    for f in out:
        f.file = rel
    return out

# ---- public --------------------------------------------------------
def run(self) -> "Scanner":
    self.findings = []
    for path in self._iter_files():
        if path.endswith(".py"):
            self.findings.extend(self._scan_python(path))
        elif path.endswith(".sql"):
            self.findings.extend(self._scan_sql(path))
    # stable sort: severity desc, confidence desc, file
    self.findings.sort(key=lambda f: (-f.severity, -f.confidence, f.file, f.line))
    return self

def to_dataframe(self):
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required for to_dataframe(); pip install pandas")
    rows = []
    for f in self.findings:
        d = f.to_dict()
        d.pop("signals", None)
        d.pop("mitigation_template", None)
        rows.append(d)
    return pd.DataFrame(rows)

def to_json(self, path: Optional[str] = None) -> str:
    payload = [f.to_dict() for f in self.findings]
    text = json.dumps(payload, indent=2, default=str)
    if path:
        Path(path).write_text(text)
    return text

def summary(self):
    sev = Counter(f.severity for f in self.findings)
    lab = Counter(f.confidence_label for f in self.findings)
    cwe = Counter(f.cwe for f in self.findings)
    print(f"Scanned: {self.root}")
    print(f"Total findings: {len(self.findings)}")
    print()
    print("By severity:")
    for s in sorted(sev, reverse=True):
        print(f"  V{s}: {sev[s]}")
    print()
    print("By confidence:")
    for k in ("likely_real", "needs_review", "likely_false_positive"):
        print(f"  {k:>25}: {lab.get(k, 0)}")
    print()
    print("Top CWEs:")
    for c, n in cwe.most_common(10):
        print(f"  {c}: {n}")

def explain(self, finding_or_row):
    """Print signal breakdown for a finding (accepts Finding or DataFrame row)."""
    if hasattr(finding_or_row, "signals"):
        f = finding_or_row
    else:
        # DataFrame row — match by file + line + cwe
        match = [
            x for x in self.findings
            if x.file == finding_or_row["file"]
            and x.line == finding_or_row["line"]
            and x.cwe == finding_or_row["cwe"]
        ]
        if not match:
            print("No matching finding.")
            return
        f = match[0]
    print(f"{f.cwe} ({f.name}) at {f.file}:{f.line}")
    print(f"  severity: V{f.severity}   confidence: {f.confidence:.2f} ({f.confidence_label})")
    if f.description:
        import textwrap
        print("  description:")
        for line in textwrap.wrap(f.description, width=76):
            print(f"    {line}")
    if f.impact:
        import textwrap
        print("  impact:")
        for line in textwrap.wrap(f.impact, width=76):
            print(f"    {line}")
    print(f"  snippet:  {f.snippet}")
    print(f"  finding:  {f.message}")
    print("  signals:")
    for note, delta in f.signals:
        sign = "+" if delta >= 0 else ""
        print(f"    {sign}{delta:+.2f}  {note}")
    if f.suggested_fix:
        print("  fix:")
        for line in f.suggested_fix.splitlines():
            print(f"    {line}")
    if f.mitigation_template:
        print("  mitigation template (if FP):")
        for line in f.mitigation_template.splitlines():
            print(f"    {line}")
```

# —————————————————————————

# CLI

# —————————————————————————

def _cli(argv=None):
p = argparse.ArgumentParser(description=“Pre-flight Veracode-style scanner for Python + SnowSQL.”)
p.add_argument(“path”, help=“Root directory to scan.”)
p.add_argument(”–json”, help=“Write findings to JSON file.”)
p.add_argument(”–min-severity”, type=int, default=1, help=“1..5 (Veracode V1..V5)”)
p.add_argument(”–likely-real-only”, action=“store_true”,
help=“Only show likely_real (confidence >= 0.70).”)
p.add_argument(”–no-tests”, action=“store_true”, help=“Skip test files.”)
args = p.parse_args(argv)

```
s = Scanner(args.path, include_tests=not args.no_tests).run()
findings = [f for f in s.findings if f.severity >= args.min_severity]
if args.likely_real_only:
    findings = [f for f in findings if f.confidence_label == "likely_real"]

s.summary()
print()
print(f"Showing {len(findings)} findings"
      f"{' (likely_real only)' if args.likely_real_only else ''}:")
print()
import textwrap
for f in findings:
    print(f"  V{f.severity}  conf={f.confidence:.2f} [{f.confidence_label:>22}]  "
          f"{f.cwe} ({f.name})  {f.file}:{f.line}")
    print(f"        > {f.snippet}")
    if f.description:
        for line in textwrap.wrap(f.description, width=78,
                                  initial_indent="        What:    ",
                                  subsequent_indent="                 "):
            print(line)
    if f.impact:
        for line in textwrap.wrap(f.impact, width=78,
                                  initial_indent="        Impact:  ",
                                  subsequent_indent="                 "):
            print(line)
    print(f"        Finding: {f.message}")
    print()

if args.json:
    s.to_json(args.json)
    print(f"Wrote: {args.json}")
return 0
```

if **name** == “**main**”:
sys.exit(_cli())
