from engine.gries import analyze_logic

cases = {
    "CORRECT": "n = 10\ntotal = 0\ni = 1\nwhile i <= n:\n    total += i\n    i += 1\nprint(total)\n",
    "S-ERROR(total=1)": "n = 10\ntotal = 1\ni = 1\nwhile i <= n:\n    total += i\n    i += 1\nprint(total)\n",
    "S-ERROR(fact=0)": "n = 5\nfact = 0\ni = 1\nwhile i <= n:\n    fact *= i\n    i += 1\nprint(fact)\n",
    "T-ERROR(backwards)": "n = 10\ntotal = 0\ni = 1\nwhile i <= n:\n    total += i\n    i -= 1\nprint(total)\n",
    "T-ERROR(missing i update)": "n = 5\nfact = 1\ni = 1\nwhile i <= n:\n    fact *= i\nprint(fact)\n",
    "E-ERROR(print i)": "n = 10\ntotal = 0\ni = 1\nwhile i <= n:\n    total += i\n    i += 1\nprint(i)\n",
    "E-ERROR(no print)": "num = 12345\ncount = 0\nwhile num > 0:\n    num = num // 10\n    count += 1\n",
    "E-ERROR(while True)": "n = 5\ni = n\nwhile True:\n    print(i)\n    i -= 1\n",
    "S-ERROR(undefined count)": "n = 5\nwhile count < n:\n    count += 1\nprint(count)\n",
    "countdown correct": "n = 5\ni = n\nwhile i >= 1:\n    print(i)\n    i -= 1\n",
}

for name, code in cases.items():
    r = analyze_logic(code)
    print("=" * 64)
    print(name, "-> ok:", r["ok"], "| error states:", r.get("states"))
    for iss in r["issues"]:
        print(f"  [{iss['severity']}] {iss['state']} L{iss['line']}: {iss['message'][:100]}")
    if r["corrected_code"]:
        print("  -- corrected --")
        print("\n".join("    " + l for l in r["corrected_code"].splitlines()))
