"""
Gries-state analysis engine for CodeLens AI.

Implements the ideas of the base paper (Leyton-Yela et al., 2026):
  * WhileLoopFinder  -> partitions Python source into the three Gries states:
        S (Initial state)  : statements before the while loop (precondition)
        T (Transformation) : while-loop body (invariant / state update)
        E (Final state)    : loop condition + statements after the loop
  * AST feature extraction for each state (used by the ML classifiers, just
    like the paper feeds the Gries partitions to the learning models).
  * Rule-based logic-error detection localised to S / T / E, with automatic
    correction suggestions.
"""
import ast
import builtins
import re

BUILTIN_NAMES = set(dir(builtins)) | {"self", "cls"}

# ---------------------------------------------------------------------------
# WhileLoopFinder (paper Section II-A): locate the while loop and split states
# ---------------------------------------------------------------------------
class WhileLoopFinder:
    """Finds the while-loop node and returns the Gries S / T / E partitions."""

    def find(self, tree):
        # 1) module-level while
        wh = None
        pre, post = [], []
        for idx, node in enumerate(getattr(tree, "body", [])):
            if isinstance(node, ast.While):
                wh = node
                pre = tree.body[:idx]
                post = tree.body[idx + 1:]
                break
        if wh is None:
            # 2) first while anywhere (e.g. inside a function / if)
            for node in ast.walk(tree):
                if isinstance(node, ast.While):
                    wh = node
                    break
        return wh, pre, post

    def regions(self, source):
        tree = ast.parse(source)
        wh, pre, post = self.find(tree)
        if wh is None:
            return None
        return {
            "tree": tree,
            "while": wh,
            "S": pre,                       # initial state nodes
            "T": list(wh.body),             # transformation state nodes
            "E_test": wh.test,              # final state: loop condition
            "E_post": post,                 # final state: statements after loop
            "S_lines": self._span(pre),
            "T_lines": (wh.body[0].lineno, wh.body[-1].end_lineno) if wh.body else (wh.lineno, wh.lineno),
            "E_lines": self._span([wh.test] + post, default=(wh.lineno, wh.lineno)),
        }

    @staticmethod
    def _span(nodes, default=(0, 0)):
        lines = []
        for n in nodes:
            if hasattr(n, "lineno"):
                lines.append((n.lineno, getattr(n, "end_lineno", n.lineno)))
        if not lines:
            return default
        return (min(a for a, _ in lines), max(b for _, b in lines))


# ---------------------------------------------------------------------------
# AST feature extraction (per Gries state) -> feature vector for the ML models
# ---------------------------------------------------------------------------
AUG_OPS = {ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mult",
           ast.Div: "Div", ast.FloorDiv: "Floor", ast.Mod: "Mod"}
BIN_OPS = {ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mult",
           ast.Div: "Div", ast.FloorDiv: "Floor", ast.Mod: "Mod", ast.Pow: "Pow"}
CMP_OPS = {ast.Lt: "Lt", ast.LtE: "LtE", ast.Gt: "Gt", ast.GtE: "GtE",
           ast.Eq: "Eq", ast.NotEq: "NotEq"}

REGION_FEATURES = [
    "lines", "Assign", "AugAdd", "AugSub", "AugMult", "AugDiv",
    "BinAdd", "BinSub", "BinMult", "BinDiv", "BinFloor", "BinMod", "BinPow",
    "CmpLt", "CmpLtE", "CmpGt", "CmpGtE", "CmpEq", "CmpNotEq",
    "Bool", "If", "Call", "Print", "Subscript", "Const", "Name", "Break",
]
FEATURE_NAMES = (
    [f"S_{f}" for f in REGION_FEATURES]
    + [f"T_{f}" for f in REGION_FEATURES]
    + [f"E_{f}" for f in REGION_FEATURES]
    + ["Total_lines", "While_count", "For_count", "Func_count"]
)


def _count_region(nodes):
    c = {f: 0 for f in REGION_FEATURES}
    min_line, max_line = None, None
    for n in nodes:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Assign):
                c["Assign"] += 1
            elif isinstance(sub, ast.AugAssign):
                name = AUG_OPS.get(type(sub.op))
                if name == "Add":
                    c["AugAdd"] += 1
                elif name == "Sub":
                    c["AugSub"] += 1
                elif name == "Mult":
                    c["AugMult"] += 1
                elif name in ("Div", "Floor", "Mod"):
                    c["AugDiv"] += 1
            elif isinstance(sub, ast.BinOp):
                name = BIN_OPS.get(type(sub.op))
                if name:
                    c[f"Bin{name}"] += 1
            elif isinstance(sub, ast.Compare):
                for op in sub.ops:
                    name = CMP_OPS.get(type(op))
                    if name:
                        c[f"Cmp{name}"] += 1
            elif isinstance(sub, ast.BoolOp):
                c["Bool"] += 1
            elif isinstance(sub, ast.If):
                c["If"] += 1
            elif isinstance(sub, ast.Call):
                c["Call"] += 1
                if isinstance(sub.func, ast.Name) and sub.func.id == "print":
                    c["Print"] += 1
            elif isinstance(sub, ast.Subscript):
                c["Subscript"] += 1
            elif isinstance(sub, ast.Constant):
                c["Const"] += 1
            elif isinstance(sub, ast.Name):
                c["Name"] += 1
            elif isinstance(sub, ast.Break):
                c["Break"] += 1
            if hasattr(sub, "lineno"):
                lo = sub.lineno
                hi = getattr(sub, "end_lineno", sub.lineno)
                min_line = lo if min_line is None else min(min_line, lo)
                max_line = hi if max_line is None else max(max_line, hi)
    c["lines"] = (max_line - min_line + 1) if min_line else 0
    return c


def extract_features(source):
    """Return (feature_dict, regions) for Python source. Raises SyntaxError."""
    reg = WhileLoopFinder().regions(source)
    if reg is None:
        # code without a while loop: treat everything as T
        tree = ast.parse(source)
        reg = {"tree": tree, "while": None,
               "S": [], "T": list(tree.body), "E_test": None, "E_post": [],
               "S_lines": (0, 0), "T_lines": (1, source.count("\n") + 1),
               "E_lines": (0, 0)}
    s = _count_region(reg["S"])
    t = _count_region(reg["T"])
    e_nodes = ([reg["E_test"]] if reg["E_test"] is not None else []) + reg["E_post"]
    e = _count_region(e_nodes)

    feats = {}
    feats.update({f"S_{k}": v for k, v in s.items()})
    feats.update({f"T_{k}": v for k, v in t.items()})
    feats.update({f"E_{k}": v for k, v in e.items()})
    total_lines = source.count("\n") + 1
    feats["Total_lines"] = total_lines
    feats["While_count"] = sum(isinstance(n, ast.While) for n in ast.walk(reg["tree"]))
    feats["For_count"] = sum(isinstance(n, ast.For) for n in ast.walk(reg["tree"]))
    feats["Func_count"] = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                              for n in ast.walk(reg["tree"]))
    return feats, reg


def feature_vector(source):
    feats, _ = extract_features(source)
    return [feats.get(name, 0) for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# AST textual tree (shown on the analysis page)
# ---------------------------------------------------------------------------
def ast_tree_text(source):
    tree = ast.parse(source)
    lines = []

    def node_name(n):
        label = type(n).__name__
        extra = ""
        if isinstance(n, ast.Name):
            extra = f"  {n.id}"
        elif isinstance(n, ast.Constant):
            extra = f"  {n.value!r}"
        elif isinstance(n, ast.While):
            extra = "  (loop)"
        elif isinstance(n, ast.Compare):
            extra = "  " + " / ".join(type(o).__name__ for o in n.ops)
        elif isinstance(n, ast.AugAssign):
            extra = "  " + type(n.op).__name__
        elif isinstance(n, ast.BinOp):
            extra = "  " + type(n.op).__name__
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            extra = f"  {n.id}" if hasattr(n, "id") else f"  {n.func.id}()"
        return f"{label}{extra}"

    def walk(n, depth, state):
        indent = "  " * depth
        tag = f"[{state}] " if state else ""
        marker = ""
        if isinstance(n, ast.While):
            marker = "  <-- Gries partition boundary"
        lines.append(f"{indent}{tag}{node_name(n)}{marker}  (line {getattr(n, 'lineno', '?')})")
        for child in ast.iter_child_nodes(n):
            child_state = state
            if state == "S" and isinstance(n, ast.While):
                child_state = "T"
            if state == "T" and isinstance(n, ast.Module):
                child_state = "E"
            walk(child, depth + 1, child_state)

    # mark states: walk module manually
    lines.append("Module")
    reg = WhileLoopFinder().regions(source) if any(isinstance(x, ast.While) for x in ast.walk(tree)) else None
    if reg is None:
        for n in tree.body:
            walk(n, 1, "T")
        return "\n".join(lines)
    for n in reg["S"]:
        walk(n, 1, "S")
    wh = reg["while"]
    lines.append(f"  [E] While  (line {wh.lineno}) condition: {ast.unparse(wh.test)}")
    for n in wh.body:
        walk(n, 2, "T")
    for n in reg["E_post"]:
        walk(n, 1, "E")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logic-error detector (rule based, localised to Gries states) + corrector
# ---------------------------------------------------------------------------
SUM_NAMES = ("total", "sum", "suma", "count", "cnt", "score", "average", "avg")
MUL_NAMES = ("fact", "product", "prod", "result", "res", "power", "lcm")


class LogicReport:
    def __init__(self):
        self.issues = []          # {state, severity, line, message, fix}
        self.edits = []           # (line_no_1based, kind, text) applied to source

    def add(self, state, severity, line, message, fix=None):
        self.issues.append({"state": state, "severity": severity,
                            "line": line, "message": message, "fix": fix})

    @property
    def has_errors(self):
        return any(i["severity"] == "error" for i in self.issues)


def _assigned_names(tree):
    defined = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        elif isinstance(n, (ast.For, ast.comprehension)):
            targets = [n.target]
        elif isinstance(n, ast.NamedExpr):
            targets = [n.target]
        else:
            targets = []
        for tgt in targets:
            for t in ast.walk(tgt):
                if isinstance(t, ast.Name):
                    defined.add(t.id)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(n.name)
            for a in n.args.args + n.args.kwonlyargs:
                defined.add(a.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                defined.add((a.asname or a.name).split(".")[0])
    return defined


def _used_names(tree):
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)
    return used


def _init_values(pre_nodes):
    """name -> (value_or_None, lineno, raw_line_node) for simple assignments."""
    vals = {}
    for n in pre_nodes:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            val = None
            if isinstance(n.value, ast.Constant):
                val = n.value.value
            vals[name] = (val, n.lineno, n)
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            vals.setdefault(n.target.id, (None, n.lineno, n))
    return vals


def analyze_logic(source):
    """
    Returns dict:
      ok (bool), issues list, corrected_code (str|None), regions info,
      detected_states set of states containing errors.
    """
    report = LogicReport()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        rep = LogicReport()
        rep.add("?", "error", e.lineno or 0,
                f"Syntax error: {e.msg}. The paper requires syntactically valid "
                f"code so that AST / Gries-state analysis can run.", None)
        return {"ok": False, "issues": rep.issues, "corrected_code": None,
                "syntax_error": True, "states": []}

    reg = WhileLoopFinder().regions(source)
    src_lines = source.splitlines()

    # --- undefined names (S: missing initialization / typo) ----------------
    defined = _assigned_names(tree) | BUILTIN_NAMES
    undefined = _used_names(tree) - defined
    for name in sorted(undefined):
        # find first use line
        line_no = 0
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load):
                line_no = n.lineno
                break
        report.add("S", "error", line_no,
                   f"Name '{name}' is used but never initialized before the loop "
                   f"(Initial-state S error): the precondition is incomplete.",
                   None)

    if reg is None:
        return {"ok": not report.has_errors, "issues": report.issues,
                "corrected_code": None, "syntax_error": False, "states": [],
                "no_while": True}

    wh = reg["while"]
    init_vals = _init_values(reg["S"])
    body_assign = {}
    for n in ast.walk(ast.Module(body=reg["T"], type_ignores=[])):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            body_assign.setdefault(n.target.id, []).append(n)
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    body_assign.setdefault(t.id, []).append(n)

    # --- E: loop condition analysis ----------------------------------------
    compares = [n for n in ast.walk(wh.test) if isinstance(n, ast.Compare)]
    test_names = {n.id for n in ast.walk(wh.test) if isinstance(n, ast.Name)}
    body_assigned_names = set(body_assign.keys())

    is_const_test = isinstance(wh.test, ast.Constant)
    if is_const_test and wh.test.value is True and not any(isinstance(n, ast.Break) for n in ast.walk(wh)):
        # try to reconstruct a condition from the driver + bound
        driver, inc = None, None
        for nm in test_names | body_assigned_names:
            augs = [a for a in body_assign.get(nm, []) if isinstance(a, ast.AugAssign)]
            if augs and isinstance(augs[-1].op, (ast.Add, ast.Sub)):
                driver = nm
                inc = isinstance(augs[-1].op, ast.Add)
                break
        bound = None
        for nm, (val, ln, _) in init_vals.items():
            if nm != driver and nm not in body_assigned_names:
                bound = nm
                break
        fix = None
        if driver:
            cond = f"{driver} <= {bound}" if (inc and bound) else (f"{driver} >= 1" if not inc else f"{driver} < {bound or 'n'}")
            fix = ("replace_line", wh.lineno, f"while {cond}:")
        report.add("E", "error", wh.lineno,
                   "Loop condition is 'while True' with no break -> the final "
                   "state E is never reached (infinite loop).", fix)

    for cmp in compares:
        left_names = {n.id for n in ast.walk(cmp.left) if isinstance(n, ast.Name)}
        for op, comp in zip(cmp.ops, cmp.comparators):
            right_names = {n.id for n in ast.walk(comp) if isinstance(n, ast.Name)}
            driver = next((n for n in (left_names | right_names) if n in body_assigned_names), None)
            if driver is None:
                continue
            augs = [a for a in body_assign.get(driver, []) if isinstance(a, ast.AugAssign)]
            if not augs:
                continue
            aug = augs[-1]
            # wrong update direction -> infinite loop (T)
            if isinstance(op, (ast.Lt, ast.LtE)) and isinstance(aug.op, ast.Sub):
                new_line = src_lines[aug.lineno - 1].replace("-=", "+=", 1)
                report.add("T", "error", aug.lineno,
                           f"Counter '{driver}' moves backwards ({ast.unparse(aug).strip()}) "
                           f"while the condition expects it to increase: the state "
                           f"transformation T never converges (infinite loop).",
                           ("replace_line", aug.lineno, new_line))
            elif isinstance(op, (ast.Gt, ast.GtE)) and isinstance(aug.op, ast.Add):
                new_line = src_lines[aug.lineno - 1].replace("+=", "-=", 1)
                report.add("T", "error", aug.lineno,
                           f"Counter '{driver}' moves forwards ({ast.unparse(aug).strip()}) "
                           f"while the condition expects it to decrease: the state "
                           f"transformation T never converges (infinite loop).",
                           ("replace_line", aug.lineno, new_line))
            # step off-by-one i += 2 on a <= counter loop
            elif isinstance(aug.op, ast.Add) and isinstance(aug.value, ast.Constant) and aug.value.value == 2 \
                    and isinstance(op, (ast.Lt, ast.LtE)):
                new_line = re.sub(r"\+=\s*2", "+= 1", src_lines[aug.lineno - 1])
                report.add("T", "warning", aug.lineno,
                           f"'{driver}' advances by 2 on each iteration, so values are "
                           f"skipped (possible transformation-state T error).",
                           ("replace_line", aug.lineno, new_line))
            # strict boundary warning (E)
            if isinstance(op, ast.Lt) and isinstance(aug.op, ast.Add):
                report.add("E", "warning", wh.lineno,
                           f"Condition uses strict '<' — check the loop boundary: "
                           f"if the task includes the last value (e.g. 1..N), it "
                           f"should be '<=' (final-state E off-by-one).", None)

    # --- driver never updated inside the loop (T) --------------------------
    def _driver_candidate(names):
        """Pick the variable most likely to be the loop counter/driver."""
        def score(nm):
            s = 0
            if nm in ("i", "j", "k", "count", "cnt", "step"):
                s += 10
            if nm in test_names and nm in {x.id for x in ast.walk(wh.test) if isinstance(x, ast.Name)}:
                s += 1
            augs = [a for a in body_assign.get(nm, []) if isinstance(a, ast.AugAssign)]
            if augs and isinstance(augs[-1].op, ast.Add):
                s += 5
            if nm in init_vals:
                val = init_vals[nm][0]
                if val == 1:
                    s += 3
                elif isinstance(val, int):
                    s += 1
            return s
        return max(sorted(names), key=score) if names else None

    for cmp in compares:
        cmp_names = {x.id for x in ast.walk(cmp) if isinstance(x, ast.Name)}
        if cmp_names & body_assigned_names:
            continue  # this comparison already has a driver updated in the body
        cand = _driver_candidate(cmp_names)
        if cand is None:
            continue
        inc = any(isinstance(op, (ast.Lt, ast.LtE)) for op in cmp.ops)
        indent = re.match(r"\s*", src_lines[wh.body[-1].lineno - 1]).group(0)
        sign = "+=" if inc else "-="
        report.add("T", "error", wh.body[-1].end_lineno,
                   f"Loop driver '{cand}' appears in the condition but is never "
                   f"updated in the loop body: transformation state T is missing "
                   f"(infinite loop).",
                   ("insert_after", wh.body[-1].end_lineno, f"{indent}{cand} {sign} 1"))

    # --- S / T: accumulator initialization & updates -----------------------
    for nm, nodes in body_assign.items():
        augs = [a for a in nodes if isinstance(a, ast.AugAssign)]
        if not augs:
            continue
        aug = augs[-1]
        low = nm.lower()
        # missing initialization
        if nm not in init_vals:
            if nm not in undefined:
                start = "1" if isinstance(aug.op, ast.Mult) else "0"
                indent = re.match(r"\s*", src_lines[wh.lineno - 1]).group(0)
                report.add("S", "error", wh.lineno,
                           f"Accumulator '{nm}' is updated in the loop but never "
                           f"initialized before it (initial-state S error).",
                           ("insert_before", wh.lineno, f"{indent}{nm} = {start}"))
            continue
        val, ln, _ = init_vals[nm]
        if isinstance(aug.op, ast.Add) and val == 1 and any(k in low for k in SUM_NAMES):
            new_line = re.sub(r"=\s*1\s*$", "= 0", src_lines[ln - 1])
            report.add("S", "error", ln,
                       f"Summation variable '{nm}' starts at 1; it must start at 0 "
                           f"(initial-state S error: the precondition is wrong).",
                       ("replace_line", ln, new_line))
        elif isinstance(aug.op, ast.Mult) and val == 0:
            new_line = re.sub(r"=\s*0\s*$", "= 1", src_lines[ln - 1])
            report.add("S", "error", ln,
                       f"Multiplicative variable '{nm}' starts at 0, so every result "
                       f"stays 0; it must start at 1 (initial-state S error).",
                       ("replace_line", ln, new_line))
        elif isinstance(aug.op, ast.Sub) and any(k in low for k in SUM_NAMES):
            new_line = src_lines[aug.lineno - 1].replace("-=", "+=", 1)
            report.add("T", "error", aug.lineno,
                       f"'{nm}' accumulates with '-=' but its name implies a running "
                       f"sum/count: the transformation T updates the state in the wrong "
                       f"direction.", ("replace_line", aug.lineno, new_line))
        elif isinstance(aug.op, ast.Add) and any(k in low for k in MUL_NAMES) and val is not None and val <= 1:
            new_line = src_lines[aug.lineno - 1].replace("+=", "*=", 1)
            report.add("T", "error", aug.lineno,
                       f"'{nm}' is a product/result variable but accumulates with '+=' "
                       f"instead of '*=' (transformation-state T error).",
                       ("replace_line", aug.lineno, new_line))

    # --- S: boolean flag initialized False but required True ----------------
    for cmp_node in [wh.test] + [c for c in compares]:
        for n in ast.walk(cmp_node):
            if isinstance(n, ast.Name) and n.id in init_vals:
                val, ln, _ = init_vals[n.id]
                if val is False and n.id in body_assigned_names:
                    new_line = re.sub(r"=\s*False\s*$", "= True", src_lines[ln - 1])
                    report.add("S", "error", ln,
                               f"Flag '{n.id}' gates the loop but starts as False, so the "
                               f"loop never runs (initial-state S error).",
                               ("replace_line", ln, new_line))

    # --- E: result after the loop (print / final value) --------------------
    acc_names = [nm for nm, nodes in body_assign.items()
                 if any(isinstance(a, ast.AugAssign) for a in nodes)
                 and nm not in test_names]
    prints = [n for n in ast.walk(ast.Module(body=reg["E_post"], type_ignores=[]))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
    body_has_print = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
        for n in ast.walk(ast.Module(body=reg["T"], type_ignores=[])))
    if acc_names and not reg["E_post"] and not body_has_print:
        indent = re.match(r"\s*", src_lines[wh.lineno - 1]).group(0)
        after = wh.body[-1].end_lineno
        report.add("E", "error", after,
                   f"Nothing is reported after the loop: the final state E is missing "
                   f"a result (e.g. print({acc_names[0]})).",
                   ("insert_after", after, f"{indent}print({acc_names[0]})"))
    elif acc_names and prints:
        p = prints[0]
        printed_names = {x.id for x in ast.walk(p) if isinstance(x, ast.Name)}
        if not (printed_names & set(acc_names)):
            new_line = re.sub(r"print\(.*\)", f"print({acc_names[0]})",
                              src_lines[p.lineno - 1])
            report.add("E", "error", p.lineno,
                       f"The program prints {', '.join(sorted(printed_names)) or 'a value'} "
                       f"instead of the computed result '{acc_names[0]}' (final-state E "
                       f"error: wrong output).",
                       ("replace_line", p.lineno, new_line))

    # --- apply edits --------------------------------------------------------
    corrected = None
    error_edits = [i for i in report.issues
                   if i["severity"] == "error" and i["fix"]]
    if error_edits:
        lines = list(src_lines)
        edits = sorted((i["fix"] for i in error_edits),
                       key=lambda e: e[1], reverse=True)
        for kind, ln, text in edits:
            if kind == "replace_line":
                if 1 <= ln <= len(lines):
                    lines[ln - 1] = text
            elif kind == "insert_before":
                lines.insert(ln - 1, text)
            elif kind == "insert_after":
                lines.insert(ln, text)
        candidate = "\n".join(lines)
        try:
            ast.parse(candidate)
            corrected = candidate
        except SyntaxError:
            corrected = None

    states = sorted({i["state"] for i in report.issues if i["severity"] == "error"})
    return {
        "ok": not report.has_errors,
        "issues": report.issues,
        "corrected_code": corrected,
        "syntax_error": False,
        "states": states,
        "regions": {
            "S": reg["S_lines"], "T": reg["T_lines"], "E": reg["E_lines"],
        },
    }
