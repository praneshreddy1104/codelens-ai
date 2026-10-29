"""
Dataset builder for CodeLens AI.

BASE PAPER:
  Leyton-Yela, Mora, Insuasti, Barco-Jimenez (2026),
  "Machine Learning Models for Automatic Classification of Programming Tasks
   in Python", IEEE Revista Iberoamericana de Tecnologias del Aprendizaje.

DATA SOURCES (taken ONLY from the base paper's references, last page):
  [25] IBM Project CodeNet  - https://github.com/IBM/ProjectCodeNet
  [26] Codeforwin           - https://codeforwin.org/
  [27] PYnative             - https://pynative.com/

The paper compiled 70 introductory "while-loop" Python exercises from those
repositories, then augmented them (GPT data-augmentation prompt, Prompt 1 in
the paper) to 3,000 records.  This module reproduces the *same dataset design
described in the paper* entirely offline:

  CSV columns (paper Section II-A):
    Description  - English problem description
    Solution     - Python solution (with a logic error)
    Feedback     - tutor feedback for the incorrect solution
    Label        - one of the 7 Gries labels
    Label_Code   - numeric code (paper Section II-B):
                     ys  = 0  Initial state
                     ye  = 1  Final state
                     yt  = 2  State transformation
                     yst = 3  Initial state + State transformation
                     yse = 4  Initial state + Final state
                     yet = 5  Final state + State transformation
                     yste= 6  Initial + Final + Transformation

  Label distribution (paper Section II-A):
    29% S-type, 29% E-type, 29% T-type, ~4% each pair, ~3% all three.

Code is partitioned into Gries states exactly like the paper's WhileLoopFinder:
  S = lines before the while loop (initialization / precondition)
  E = loop condition + lines after the loop (postcondition / final state)
  T = while-loop body (state transformation / invariant)
Errors are *logic* errors only (no syntax errors), following Prompt 1 rules:
  S errors misconfigure initialization, T errors break the update inside the
  loop, E errors break the loop condition or the final output.
"""
import ast
import csv
import random
import re

RNG = random.Random(2023)  # paper: seed 2023 for every random step

# ---------------------------------------------------------------------------
# 20 base while-loop exercises, in the style of Codeforwin [26] / PYnative [27]
# (classic beginner loop tasks listed in those repositories).
# Each problem stores its Gries sections separately so that mutations can be
# injected into exactly one state (S / T / E), as the paper's Prompt 1 demands.
# ---------------------------------------------------------------------------
PROBLEMS = [
    {
        "id": 1, "phrase": "find the sum of the first N natural numbers",
        "init": ["n = 10", "total = 0", "i = 1"],
        "cond": "i <= n",
        "body": ["total += i", "i += 1"],
        "post": ["print(total)"],
        "driver": "i", "acc": "total", "acc_kind": "add",
    },
    {
        "id": 2, "phrase": "compute the factorial of a number N",
        "init": ["n = 5", "fact = 1", "i = 1"],
        "cond": "i <= n",
        "body": ["fact *= i", "i += 1"],
        "post": ["print(fact)"],
        "driver": "i", "acc": "fact", "acc_kind": "mul",
    },
    {
        "id": 3, "phrase": "count the number of digits in an integer",
        "init": ["num = 12345", "count = 0"],
        "cond": "num > 0",
        "body": ["num = num // 10", "count += 1"],
        "post": ["print(count)"],
        "driver": "num", "acc": "count", "acc_kind": "add",
    },
    {
        "id": 4, "phrase": "reverse the digits of a number",
        "init": ["num = 1234", "rev = 0"],
        "cond": "num > 0",
        "body": ["digit = num % 10", "rev = rev * 10 + digit", "num = num // 10"],
        "post": ["print(rev)"],
        "driver": "num", "acc": "rev", "acc_kind": "acc_assign",
    },
    {
        "id": 5, "phrase": "calculate the sum of the digits of a number",
        "init": ["num = 1234", "total = 0"],
        "cond": "num > 0",
        "body": ["total += num % 10", "num = num // 10"],
        "post": ["print(total)"],
        "driver": "num", "acc": "total", "acc_kind": "add_expr",
    },
    {
        "id": 6, "phrase": "print numbers from N down to 1 using a while loop",
        "init": ["n = 5", "i = n"],
        "cond": "i >= 1",
        "body": ["print(i)", "i -= 1"],
        "post": [],
        "driver": "i", "acc": None, "acc_kind": None,
    },
    {
        "id": 7, "phrase": "print the multiplication table of a number",
        "init": ["n = 7", "i = 1"],
        "cond": "i <= 10",
        "body": ["print(n * i)", "i += 1"],
        "post": [],
        "driver": "i", "acc": None, "acc_kind": None,
    },
    {
        "id": 8, "phrase": "compute base raised to the power exp",
        "init": ["base = 2", "exp = 5", "result = 1", "i = 1"],
        "cond": "i <= exp",
        "body": ["result *= base", "i += 1"],
        "post": ["print(result)"],
        "driver": "i", "acc": "result", "acc_kind": "mul",
    },
    {
        "id": 9, "phrase": "check whether a number is prime",
        "init": ["n = 29", "is_prime = True", "i = 2"],
        "cond": "i < n and is_prime",
        "body": ["if n % i == 0:", "    is_prime = False", "i += 1"],
        "post": ["print(is_prime)"],
        "driver": "i", "acc": "is_prime", "acc_kind": "bool",
    },
    {
        "id": 10, "phrase": "print the first N Fibonacci numbers",
        "init": ["n = 10", "a = 0", "b = 1", "i = 1"],
        "cond": "i <= n",
        "body": ["print(a)", "temp = a", "a = b", "b = temp + b", "i += 1"],
        "post": [],
        "driver": "i", "acc": None, "acc_kind": None,
    },
    {
        "id": 11, "phrase": "find the greatest common divisor of two numbers",
        "init": ["a = 48", "b = 18"],
        "cond": "b != 0",
        "body": ["temp = b", "b = a % b", "a = temp"],
        "post": ["print(a)"],
        "driver": "b", "acc": None, "acc_kind": None,
    },
    {
        "id": 12, "phrase": "check whether a number is an Armstrong number",
        "init": ["num = 153", "temp = num", "total = 0"],
        "cond": "temp > 0",
        "body": ["digit = temp % 10", "total += digit ** 3", "temp = temp // 10"],
        "post": ["print(total == num)"],
        "driver": "temp", "acc": "total", "acc_kind": "add_expr",
    },
    {
        "id": 13, "phrase": "find the least common multiple of two numbers",
        "init": ["a = 4", "b = 6", "max_val = a if a > b else b", "lcm = max_val"],
        "cond": "lcm % a != 0 or lcm % b != 0",
        "body": ["lcm += max_val"],
        "post": ["print(lcm)"],
        "driver": "lcm", "acc": "lcm", "acc_kind": "add",
    },
    {
        "id": 14, "phrase": "count how many even numbers exist between 1 and N",
        "init": ["n = 20", "count = 0", "i = 1"],
        "cond": "i <= n",
        "body": ["if i % 2 == 0:", "    count += 1", "i += 1"],
        "post": ["print(count)"],
        "driver": "i", "acc": "count", "acc_kind": "add_nested",
    },
    {
        "id": 15, "phrase": "sum all even numbers between 1 and N",
        "init": ["n = 10", "total = 0", "i = 1"],
        "cond": "i <= n",
        "body": ["if i % 2 == 0:", "    total += i", "i += 1"],
        "post": ["print(total)"],
        "driver": "i", "acc": "total", "acc_kind": "add_nested",
    },
    {
        "id": 16, "phrase": "check whether a number is a palindrome",
        "init": ["num = 121", "temp = num", "rev = 0"],
        "cond": "temp > 0",
        "body": ["digit = temp % 10", "rev = rev * 10 + digit", "temp = temp // 10"],
        "post": ["print(rev == num)"],
        "driver": "temp", "acc": "rev", "acc_kind": "acc_assign",
    },
    {
        "id": 17, "phrase": "convert a decimal number to binary",
        "init": ["num = 13", "binary = 0", "place = 1"],
        "cond": "num > 0",
        "body": ["binary += (num % 2) * place", "num = num // 2", "place *= 10"],
        "post": ["print(binary)"],
        "driver": "num", "acc": "binary", "acc_kind": "add_expr",
    },
    {
        "id": 18, "phrase": "calculate the product of the digits of a number",
        "init": ["num = 1234", "product = 1"],
        "cond": "num > 0",
        "body": ["product *= num % 10", "num = num // 10"],
        "post": ["print(product)"],
        "driver": "num", "acc": "product", "acc_kind": "mul_expr",
    },
    {
        "id": 19, "phrase": "count the vowels in a string",
        "init": ["text = 'programming'", "count = 0", "i = 0"],
        "cond": "i < len(text)",
        "body": ["if text[i] in 'aeiou':", "    count += 1", "i += 1"],
        "post": ["print(count)"],
        "driver": "i", "acc": "count", "acc_kind": "add_nested",
    },
    {
        "id": 20, "phrase": "calculate the average of a list of numbers",
        "init": ["numbers = [10, 20, 30, 40]", "total = 0", "i = 0"],
        "cond": "i < len(numbers)",
        "body": ["total += numbers[i]", "i += 1"],
        "post": ["print(total / len(numbers))"],
        "driver": "i", "acc": "total", "acc_kind": "add",
    },
]

LABELS = {
    "ys":   ("Initial state", 0),
    "ye":   ("Final state", 1),
    "yt":   ("State transformation", 2),
    "yst":  ("Initial state, State transformation", 3),
    "yse":  ("Initial state, Final state", 4),
    "yet":  ("Final state, State transformation", 5),
    "yste": ("Initial state, Final state, State transformation", 6),
}

FEEDBACK = {
    "S": "Initial state error (S): the initialization before the while loop is "
         "wrong. A variable starts from an incorrect value, so the loop can "
         "never reach the correct final state. Check the assignments placed "
         "before the loop.",
    "T": "State transformation error (T): the update inside the while-loop body "
         "is missing or incorrect (counter/accumulator not updated properly), "
         "so the state fails to converge to the correct answer.",
    "E": "Final state error (E): the loop condition or the result produced "
         "after the loop is wrong. Check the while condition and the value "
         "printed/returned at the end.",
}

DESC_TEMPLATES = [
    "Write a Python program to {p}.",
    "Write a program that uses a while loop to {p}.",
    "Create Python code with a while loop to {p}.",
]

# Variable renaming augmentation (paper: paraphrase + different solutions)
RENAMES = [
    {},
    {"i": "k", "total": "suma", "count": "cnt"},
    {"num": "number", "temp": "tmp", "digit": "d"},
    {"total": "sum_val", "result": "res", "fact": "factorial"},
]


# ---------------------------------------------------------------------------
# small line helpers
# ---------------------------------------------------------------------------
def _find_line(lines, token):
    for idx, ln in enumerate(lines):
        if token in ln:
            return idx
    return -1


def _flip_augop(line):
    for a, b in [("+=", "-="), ("-=", "+="), ("//=", "*="), ("/=", "*="), ("*=", "+=")]:
        if a in line:
            return line.replace(a, b, 1)
    return line


def _flip_binop(line):
    # flip the first arithmetic operator on the right-hand side
    for a, b in [("//", "*"), ("/", "*"), ("%", "*"), ("+", "-"), ("-", "+"), ("*", "+")]:
        # only consider RHS after '='
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            if a in rhs:
                return lhs + "=" + rhs.replace(a, b, 1)
    return line


# ---------------------------------------------------------------------------
# S (Initial state) mutators  -> misconfigure initialization
# ---------------------------------------------------------------------------
def mutate_s(init, cond, body, post, p):
    init = list(init)
    acc, driver, kind = p["acc"], p["driver"], p["acc_kind"]
    choice = RNG.randrange(4)

    if choice == 0 and acc:
        # wrong starting value of the accumulator
        idx = _find_line(init, acc)
        if idx >= 0:
            line = init[idx]
            if kind in ("mul", "mul_expr"):
                init[idx] = re.sub(r"=\s*-?\d+", "= 0", line)      # factorial starts at 0
            elif kind == "bool":
                init[idx] = line.replace("True", "False")
            else:
                init[idx] = re.sub(r"=\s*-?\d+", "= 1", line)      # sum starts at 1
            return init, cond, body, post, FEEDBACK["S"]

    if choice == 1:
        # off-by-one / wrong starting value of the loop driver
        idx = _find_line(init, driver)
        if idx >= 0:
            line = init[idx]
            m = re.search(r"=\s*(\d+)\s*$", line)
            if m:
                init[idx] = line[:m.start(1)] + str(int(m.group(1)) + 1)
                return init, cond, body, post, FEEDBACK["S"]

    if choice == 2 and acc:
        # accumulator initialization missing entirely
        idx = _find_line(init, acc)
        if idx >= 0:
            del init[idx]
            return init, cond, body, post, FEEDBACK["S"]

    # choice 3 (or fallback): swap the initial values of two numeric variables
    num_lines = [(i, ln) for i, ln in enumerate(init) if re.search(r"=\s*\d+\s*$", ln)]
    if len(num_lines) >= 2:
        (i1, l1), (i2, l2) = RNG.sample(num_lines, 2)
        v1 = re.search(r"(\d+)\s*$", l1).group(1)
        v2 = re.search(r"(\d+)\s*$", l2).group(1)
        init[i1] = re.sub(r"\d+\s*$", v2, l1)
        init[i2] = re.sub(r"\d+\s*$", v1, l2)
    else:
        # fallback: bump driver literal
        idx = _find_line(init, driver)
        if idx >= 0:
            m = re.search(r"=\s*(\d+)\s*$", init[idx])
            if m:
                init[idx] = init[idx][:m.start(1)] + str(int(m.group(1)) + 1)
    return init, cond, body, post, FEEDBACK["S"]


# ---------------------------------------------------------------------------
# T (State transformation) mutators -> break the update inside the loop
# ---------------------------------------------------------------------------
def mutate_t(init, cond, body, post, p):
    body = list(body)
    acc, driver, kind = p["acc"], p["driver"], p["acc_kind"]
    didx = _find_line(body, driver)
    aidx = _find_line(body, acc) if acc else -1
    choice = RNG.randrange(6)

    if choice == 0 and didx >= 0:
        # driver update moves the wrong way -> loop never terminates
        body[didx] = _flip_augop(body[didx]) if "+=" in body[didx] or "-=" in body[didx] \
            else _flip_binop(body[didx])
        return init, cond, body, post, FEEDBACK["T"]

    if choice == 1 and didx >= 0:
        # driver update missing -> infinite loop
        del body[didx]
        return init, cond, body, post, FEEDBACK["T"]

    if choice == 2 and aidx >= 0:
        # accumulator update missing -> result never accumulates
        del body[aidx]
        return init, cond, body, post, FEEDBACK["T"]

    if choice == 3 and aidx >= 0:
        # accumulator update operator flipped
        line = body[aidx]
        if any(op in line for op in ("+=", "-=", "*=", "//=", "/=")):
            body[aidx] = _flip_augop(line)
        return init, cond, body, post, FEEDBACK["T"]

    if choice == 4 and aidx >= 0:
        # overwrites instead of accumulating: rev = rev*10+d -> rev = d
        line = body[aidx]
        lhs, sep, rhs = line.partition("=")
        if sep and acc in rhs:
            # keep only the part that does not reference the accumulator
            tokens = re.split(r"[+\-*/]", rhs)
            keep = [t.strip() for t in tokens if acc not in t and t.strip()]
            if keep:
                body[aidx] = f"{lhs}= {keep[-1]}"
                return init, cond, body, post, FEEDBACK["T"]
        # aug-assign variant: total += i  ->  total = i
        m = re.match(r"\s*(\w+)\s*\+=\s*(.+)", line)
        if m:
            body[aidx] = line.replace("+=", "=", 1)
            return init, cond, body, post, FEEDBACK["T"]

    if choice == 5 and didx >= 0 and re.search(rf"{driver}\s*\+=\s*1\s*$", body[didx]):
        # counter step off by one: i += 1 -> i += 2 (skips iterations)
        body[didx] = body[didx].replace("+= 1", "+= 2")
        return init, cond, body, post, FEEDBACK["T"]

    # fallback: flip driver update
    if didx >= 0:
        body[didx] = _flip_augop(body[didx]) if "+=" in body[didx] or "-=" in body[didx] \
            else _flip_binop(body[didx])
    return init, cond, body, post, FEEDBACK["T"]


# ---------------------------------------------------------------------------
# E (Final state) mutators -> break the loop condition or the final output
# ---------------------------------------------------------------------------
def _flip_cond(cond):
    """Off-by-one / inverted boundary in the while condition (E region)."""
    out = cond.replace("<=", "\x00").replace(">=", "\x01")
    out = out.replace("!=", "\x02").replace("==", "\x03")
    out = out.replace("<", "<=").replace(">", ">=")
    out = out.replace("\x03", "!=").replace("\x02", "==")
    out = out.replace("\x00", "<").replace("\x01", ">")
    if out == cond:   # e.g. a bare 'True' / membership test -> tautological loop
        out = "True"
    return out


def mutate_e(init, cond, body, post, p):
    post = list(post)
    acc, driver = p["acc"], p["driver"]
    choice = RNG.randrange(4)

    if choice in (0, 2):
        # wrong loop boundary (off-by-one in the while condition)
        new_cond = _flip_cond(cond)
        if new_cond != cond:
            return init, new_cond, body, post, FEEDBACK["E"]

    if choice == 1 and post:
        # final result prints/compares the wrong value
        new_post = []
        done = False
        for ln in post:
            if "print(" in ln and not done:
                if "==" in ln:                      # palindrome / armstrong check inverted
                    new_post.append(ln.replace("==", "!=", 1))
                elif acc and acc in ln:
                    new_post.append(re.sub(r"print\(.*\)", f"print({driver})", ln))
                else:
                    new_post.append(ln)
                done = True
            else:
                new_post.append(ln)
        if done:
            return init, cond, body, new_post, FEEDBACK["E"]

    if choice == 3 and post:
        # final output missing
        new_post = [ln for ln in post if "print(" not in ln]
        if len(new_post) != len(post):
            return init, cond, body, new_post, FEEDBACK["E"]

    # fallback: tautological condition -> loop never ends (condition = E region)
    return init, "True", body, post, FEEDBACK["E"]


# ---------------------------------------------------------------------------
# assemble source code + validation
# ---------------------------------------------------------------------------
def build_code(init, cond, body, post, indent="    "):
    lines = list(init)
    lines.append(f"while {cond}:")
    for b in body:
        lines.append(indent + b if not b.startswith(" ") else indent + b.lstrip())
    lines.extend(post)
    return "\n".join(lines)


def valid_solution(code):
    """Paper rule: no syntax errors; must contain exactly one while loop."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    whiles = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    return len(whiles) == 1


def apply_renames(code, mapping):
    if not mapping:
        return code
    for old, new in mapping.items():
        code = re.sub(rf"\b{old}\b", new, code)
    return code


def make_record(problem, label_key, desc_idx):
    p = problem
    init, cond, body, post = list(p["init"]), p["cond"], list(p["body"]), list(p["post"])
    fb = []

    if "s" in label_key:
        init, cond, body, post, f = mutate_s(init, cond, body, post, p); fb.append(f)
    if "t" in label_key:
        init, cond, body, post, f = mutate_t(init, cond, body, post, p); fb.append(f)
    if "e" in label_key:
        init, cond, body, post, f = mutate_e(init, cond, body, post, p); fb.append(f)

    code = build_code(init, cond, body, post)
    rename = RNG.choice(RENAMES)
    code = apply_renames(code, rename)

    if not valid_solution(code):
        return None

    desc = DESC_TEMPLATES[desc_idx].format(p=p["phrase"])
    label_name, label_code = LABELS[label_key]
    # order feedback states S -> T -> E
    fb_sorted = [x for x in (FEEDBACK["S"] if "s" in label_key else None,
                             FEEDBACK["T"] if "t" in label_key else None,
                             FEEDBACK["E"] if "e" in label_key else None) if x]
    return {
        "Description": desc,
        "Solution": code,
        "Feedback": " | ".join(fb_sorted),
        "Label": label_name,
        "Label_Code": label_code,
        "Label_Key": label_key,
    }


def build_dataset(n_total=3000, out_path=None):
    # paper distribution: 29% S, 29% T, 29% E, 4% per pair, 3% triple
    counts = {"ys": 850, "yt": 850, "ye": 850,
              "yst": 120, "yse": 120, "yet": 120, "yste": 90}
    buckets = {k: [] for k in counts}

    for p in PROBLEMS:
        for d in range(len(DESC_TEMPLATES)):
            for key in counts:
                for _ in range(60 if key == "yste" else 12):
                    rec = make_record(p, key, d)
                    if rec:
                        buckets[key].append(rec)

    rows = []
    for key, need in counts.items():
        pool = buckets[key]
        if len(pool) < need:
            # extra augmentation rounds if a bucket is short
            while len(pool) < need:
                p = RNG.choice(PROBLEMS)
                rec = make_record(p, key, RNG.randrange(len(DESC_TEMPLATES)))
                if rec:
                    pool.append(rec)
        rows.extend(RNG.sample(pool, need))

    RNG.shuffle(rows)

    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Description", "Solution", "Feedback", "Label", "Label_Code"])
            for r in rows:
                w.writerow([r["Description"], r["Solution"], r["Feedback"],
                            r["Label"], r["Label_Code"]])
    return rows


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "data", "gries_loop_tasks.csv")
    rows = build_dataset(out_path=out)
    print(f"Generated {len(rows)} records -> {out}")
    from collections import Counter
    dist = Counter(r["Label"] for r in rows)
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {v:5d}  ({v/len(rows)*100:4.1f}%)  {k}")
