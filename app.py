"""
Boolean Expression Simplification Dashboard
-------------------------------------------
Minor Project  |  Digital Logic Design

Run:
    pip install streamlit sympy pandas
    streamlit run app.py
"""

import io
import re
import itertools
from datetime import datetime

import pandas as pd
import sympy as sp
import streamlit as st
from sympy.logic.boolalg import simplify_logic

st.set_page_config(page_title="Boolean Simplifier", page_icon="🧮", layout="wide")

COLORS = ["#ff6b6b", "#4dabf7", "#51cf66", "#ffd43b",
          "#cc5de8", "#ff922b", "#20c997", "#f06595"]

PRESETS = {
    "Custom (type below)": "",
    "Classic 3-var": "A'B'C + A'BC + AB'C + ABC' + ABC",
    "Absorption": "A + AB + ABC",
    "Consensus": "AB + A'C + BC",
    "De Morgan": "(A+B)'(C+D)'",
    "XOR form": "A'B + AB'",
    "4-var": "A'B'C'D + A'B'CD + A'BC'D + A'BCD + AB'C'D + ABCD",
    "Redundant": "(A+B)(A+C)",
}


# ───────────────────────── parsing ──────────────────────────
def prime_to_not(s: str) -> str:
    """Convert postfix complement (A' and (..)') to prefix ~."""
    while "'" in s:
        i = s.index("'")
        j = i - 1
        if j < 0:
            raise ValueError("Misplaced apostrophe")
        if s[j] == ")":
            depth, k = 0, j
            while k >= 0:
                if s[k] == ")":
                    depth += 1
                elif s[k] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                k -= 1
            if k < 0:
                raise ValueError("Unbalanced brackets")
            s = s[:k] + "~" + s[k:i] + s[i + 1:]
        else:
            s = s[:j] + "~" + s[j] + s[i + 1:]
    return s


def parse_expression(text: str):
    raw = text.strip()
    if not raw:
        raise ValueError("Empty expression")
    s = raw.replace("’", "'").replace("!", "~").replace("·", "*")
    s = prime_to_not(s)
    s = s.replace("+", "|").replace("*", "&")
    s = re.sub(r"\bNOT\b", "~", s, flags=re.I)
    s = re.sub(r"\bAND\b", "&", s, flags=re.I)
    s = re.sub(r"\bOR\b", "|", s, flags=re.I)
    s = re.sub(r"\bXOR\b", "^", s, flags=re.I)
    # implicit AND:  AB  A(B+C)  (A)(B)  A~B
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"([A-Za-z]|\))\s*(?=[A-Za-z(~])", lambda m: m.group(1) + "&", s)
    letters = sorted(set(re.findall(r"[A-Za-z]", s)))
    if not letters:
        raise ValueError("No variables found")
    if len(letters) > 6:
        raise ValueError("Maximum 6 variables supported")
    s = re.sub(r"\b1\b", "true", s)
    s = re.sub(r"\b0\b", "false", s)
    local = {c: sp.Symbol(c) for c in letters}
    local.update({"true": sp.true, "false": sp.false})
    from sympy.parsing.sympy_parser import parse_expr
    expr = parse_expr(s, local_dict=local, evaluate=True)
    return expr, [local[c] for c in letters]


# ───────────────────────── core logic ───────────────────────
def truth_table(expr, vars_):
    n = len(vars_)
    rows = []
    for i in range(2 ** n):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        val = bool(expr.subs({v: bool(b) for v, b in zip(vars_, bits)}))
        rows.append((i, bits, int(val)))
    return rows


def qm_steps(n, ones, dcs):
    cur = {(m,): format(m, f"0{n}b") for m in sorted(set(ones) | set(dcs))}
    columns, primes = [], []
    while cur:
        columns.append(dict(cur))
        nxt, used = {}, set()
        for (a, pa), (b, pb) in itertools.combinations(list(cur.items()), 2):
            diff = [i for i in range(n) if pa[i] != pb[i]]
            if len(diff) == 1 and "-" not in (pa[diff[0]], pb[diff[0]]):
                d = diff[0]
                nxt[tuple(sorted(set(a) | set(b)))] = pa[:d] + "-" + pa[d + 1:]
                used.update([a, b])
        primes += [(k, p) for k, p in cur.items() if k not in used]
        cur = nxt
    return columns, primes


def lit_count(p):
    return sum(c != "-" for c in p)


def minimal_cover(primes, ones):
    ones = set(ones)
    if not ones:
        return [], [], []
    cover_of = {p: set(p[0]) & ones for p in primes}
    essential = []
    for m in sorted(ones):
        owners = [p for p in primes if m in cover_of[p]]
        if len(owners) == 1 and owners[0] not in essential:
            essential.append(owners[0])
    covered = set().union(*[cover_of[p] for p in essential]) if essential else set()
    remaining = ones - covered
    extra = []
    if remaining:
        cand = [p for p in primes if p not in essential and cover_of[p] & remaining]
        best = None
        for r in range(1, len(cand) + 1):
            for combo in itertools.combinations(cand, r):
                if remaining <= set().union(*[cover_of[p] for p in combo]):
                    cost = sum(lit_count(p[1]) for p in combo)
                    if best is None or cost < best[0]:
                        best = (cost, list(combo))
            if best:
                break
        extra = best[1] if best else []
    return essential, extra, essential + extra


def term_sop(pattern, names):
    out = ""
    for ch, v in zip(pattern, names):
        if ch == "1":
            out += v
        elif ch == "0":
            out += v + "'"
    return out or "1"


def term_pos(pattern, names):
    parts = []
    for ch, v in zip(pattern, names):
        if ch == "1":
            parts.append(v + "'")
        elif ch == "0":
            parts.append(v)
    return "(" + " + ".join(parts) + ")" if parts else "0"


def sop_string(cover, names):
    if not cover:
        return "0"
    terms = [term_sop(p[1], names) for p in cover]
    return "1" if "1" in terms else " + ".join(terms)


def pos_string(cover_of_zeros, names):
    if not cover_of_zeros:
        return "1"
    terms = [term_pos(p[1], names) for p in cover_of_zeros]
    return "0" if "0" in terms else "".join(terms)


def gray(k):
    return ["0", "1"] if k == 1 else ["00", "01", "11", "10"]


def kmap_html(n, names, ones, dcs, cover):
    rb = n // 2
    cb = n - rb
    rows, cols = gray(rb), gray(cb)
    membership = {}
    for idx, p in enumerate(cover):
        for m in p[0]:
            membership.setdefault(m, []).append(idx)
    th = "padding:8px 14px;background:#343a40;color:#fff;border:1px solid #868e96;font-family:monospace"
    h = ["<table style='border-collapse:collapse;text-align:center;margin:auto'>"]
    h.append(f"<tr><th style='{th}'>{''.join(names[:rb])} \\ {''.join(names[rb:])}</th>")
    for c in cols:
        h.append(f"<th style='{th}'>{c}</th>")
    h.append("</tr>")
    for r in rows:
        h.append(f"<tr><th style='{th}'>{r}</th>")
        for c in cols:
            m = int(r + c, 2)
            val = "1" if m in ones else ("X" if m in dcs else "0")
            groups = membership.get(m, [])
            if not groups:
                bg = "#f1f3f5"
            elif len(groups) == 1:
                bg = COLORS[groups[0] % len(COLORS)]
            else:
                step = 100 / len(groups)
                stops = ", ".join(f"{COLORS[g % len(COLORS)]} {i*step:.0f}% {(i+1)*step:.0f}%"
                                  for i, g in enumerate(groups))
                bg = f"linear-gradient(135deg, {stops})"
            weight = "700" if val == "1" else "400"
            h.append(
                f"<td style='width:70px;height:60px;border:2px solid #495057;background:{bg};"
                f"color:#212529;font-size:20px;font-weight:{weight};position:relative'>"
                f"{val}<span style='position:absolute;top:2px;right:5px;font-size:10px;color:#495057'>m{m}</span></td>"
            )
        h.append("</tr>")
    h.append("</table>")
    return "".join(h)


def circuit_dot(names, cover):
    L = ['digraph G { rankdir=LR; bgcolor="transparent"; node [fontname="Helvetica", fontsize=12]; edge [color="#868e96"];']
    for v in names:
        L.append(f'"{v}" [shape=circle, style=filled, fillcolor="#a5d8ff", label="{v}"];')
    need_not = set()
    for p in cover:
        for ch, v in zip(p[1], names):
            if ch == "0":
                need_not.add(v)
    for v in need_not:
        L.append(f'"n_{v}" [shape=invtriangle, style=filled, fillcolor="#ffe066", label="NOT", fontsize=9];')
        L.append(f'"{v}" -> "n_{v}";')
    L.append('"F" [shape=doublecircle, style=filled, fillcolor="#b2f2bb", label="F"];')
    term_nodes = []
    for i, p in enumerate(cover):
        lits = [(v if ch == "1" else f"n_{v}") for ch, v in zip(p[1], names) if ch != "-"]
        if len(lits) == 1:
            term_nodes.append(lits[0])
            continue
        node = f"AND{i+1}"
        L.append(f'"{node}" [shape=box, style="rounded,filled", fillcolor="{COLORS[i % len(COLORS)]}", label="AND\\n{term_sop(p[1], names)}"];')
        for s in lits:
            L.append(f'"{s}" -> "{node}";')
        term_nodes.append(node)
    if len(term_nodes) == 1:
        L.append(f'"{term_nodes[0]}" -> "F";')
    else:
        L.append('"OR" [shape=box, style="rounded,filled", fillcolor="#ffa8a8", label="OR"];')
        for t in term_nodes:
            L.append(f'"{t}" -> "OR";')
        L.append('"OR" -> "F";')
    L.append("}")
    return "\n".join(L)


def count_literals(expr):
    return sum(1 for a in sp.preorder_traversal(expr) if isinstance(a, sp.Symbol))


# ───────────────────────── UI ───────────────────────────────
st.markdown(
    """
    <style>
    .hero{padding:22px 28px;border-radius:16px;
          background:linear-gradient(120deg,#4c6ef5,#7950f2 55%,#e64980);color:white;margin-bottom:14px}
    .hero h1{margin:0;font-size:2rem}
    .hero p{margin:4px 0 0 0;opacity:.9}
    .ans{padding:14px 18px;border-radius:12px;border-left:6px solid #51cf66;
         background:rgba(81,207,102,.12);font-family:monospace;font-size:1.25rem}
    </style>
    <div class="hero"><h1>🧮 Boolean Expression Simplifier</h1>
    <p>Truth table • Canonical forms • Quine–McCluskey • K-Map • Logic circuit • Verification</p></div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Input")
    mode = st.radio("Input type", ["Boolean expression", "Minterms list"])
    ones, dcs, names, expr = [], [], [], None
    try:
        if mode == "Boolean expression":
            preset = st.selectbox("Examples", list(PRESETS))
            default = PRESETS[preset] or "A'B + AB' + AB"
            text = st.text_input("Expression", value=default, key=f"expr_{preset}",
                                 help="Use ' for NOT, + for OR, juxtaposition/* for AND. Brackets allowed. Example: (A+B)'C + AB")
            dc_text = st.text_input("Don't-care minterms (optional)", "", help="comma separated e.g. 3,5")
            expr, vars_ = parse_expression(text)
            names = [str(v) for v in vars_]
            n = len(names)
            tt = truth_table(expr, vars_)
            dcs = [int(x) for x in re.findall(r"\d+", dc_text)]
            ones = [i for i, _, v in tt if v == 1 and i not in dcs]
        else:
            n = st.slider("Number of variables", 2, 6, 4)
            names = list("ABCDEF"[:n])
            m_text = st.text_input("Minterms Σm", "0,1,2,5,6,7,8,9,10,14")
            d_text = st.text_input("Don't cares Σd", "")
            ones = sorted({int(x) for x in re.findall(r"\d+", m_text) if int(x) < 2 ** n})
            dcs = sorted({int(x) for x in re.findall(r"\d+", d_text) if int(x) < 2 ** n} - set(ones))
            vars_ = [sp.Symbol(c) for c in names]
            from sympy.logic.boolalg import SOPform
            tmp = SOPform(vars_, ones, dcs) if ones else sp.false
            expr = tmp
            tt = []
            for i in range(2 ** n):
                tt.append((i, [(i >> (n - 1 - k)) & 1 for k in range(n)],
                           1 if i in ones else 0))
        dcs = sorted(set(dcs) - set(ones))
        ok = True
    except Exception as e:
        ok = False
        st.error(f"Could not parse input: {e}")

if not ok:
    st.stop()

# ── computation ──
columns, primes = qm_steps(n, ones, dcs)
essential, extra, cover = minimal_cover(primes, ones)
simp_sop = sop_string(cover, names)
zeros = [i for i in range(2 ** n) if i not in ones and i not in dcs]
_, primes_z = qm_steps(n, zeros, dcs)
_, _, cover_z = minimal_cover(primes_z, zeros)
simp_pos = pos_string(cover_z, names)

canon_sop = " + ".join(term_sop(format(m, f"0{n}b"), names) for m in ones) or "0"
canon_pos = "".join(term_pos(format(m, f"0{n}b"), names) for m in zeros) or "1"

lit_orig = count_literals(expr) if mode == "Boolean expression" else len(ones) * n
lit_canon = len(ones) * n
lit_simp = sum(lit_count(p[1]) for p in cover)
red = 0 if lit_orig == 0 else round(100 * (lit_orig - lit_simp) / lit_orig)

# verification
verified = all(
    (1 if i in cover_set else 0) == (1 if i in ones else 0) or i in dcs
    for i in range(2 ** n)
    for cover_set in [set().union(*[set(p[0]) for p in cover]) if cover else set()]
)
try:
    sym_check = simplify_logic(expr, form="dnf") if n <= 6 else None
    sym_str = str(sym_check).replace("~", "¬")
except Exception:
    sym_str = "n/a"

# ── tabs ──
tabs = st.tabs(["📊 Overview", "📋 Truth Table", "🧾 Canonical Forms",
                "🔬 Quine–McCluskey", "🗺️ K-Map", "🔌 Circuit", "✅ Report"])

with tabs[0]:
    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Original**\n\n<div class='ans'>{text if mode=='Boolean expression' else 'Σm(' + ','.join(map(str, ones)) + ')'}</div>",
                unsafe_allow_html=True)
    c2.markdown(f"**Minimal SOP**\n\n<div class='ans'>F = {simp_sop}</div>", unsafe_allow_html=True)
    st.markdown(f"**Minimal POS:** `F = {simp_pos}`")
    st.write("")
    m = st.columns(5)
    m[0].metric("Variables", n)
    m[1].metric("Minterms", len(ones))
    m[2].metric("Prime implicants", len(primes))
    m[3].metric("Essential PIs", len(essential))
    m[4].metric("Literals saved", f"{lit_orig - lit_simp}", f"-{red}%")
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("**Complexity comparison (literals)**")
        df = pd.DataFrame({"Literals": [lit_orig, lit_canon, lit_simp]},
                          index=["Original", "Canonical SOP", "Minimal SOP"])
        st.bar_chart(df)
    with b:
        st.markdown("**Output distribution**")
        df2 = pd.DataFrame({"Count": [len(ones), len(zeros), len(dcs)]},
                           index=["F = 1", "F = 0", "Don't care"])
        st.bar_chart(df2)
    if verified:
        st.success("✅ Simplified expression verified against the truth table for all inputs.")
    else:
        st.error("❌ Mismatch found – check the input.")

with tabs[1]:
    st.subheader("Truth Table")
    data = []
    for i, bits, v in tt:
        row = {f"{names[k]}": bits[k] for k in range(n)}
        row["F"] = "X" if i in dcs else v
        row["Minterm"] = f"m{i}"
        data.append(row)
    dft = pd.DataFrame(data)
    st.dataframe(dft.style.map(lambda x: "background-color:#51cf6655" if x == 1 else ("background-color:#ffd43b55" if x == "X" else ""),
                               subset=["F"]), hide_index=True)
    st.download_button("⬇️ Download truth table (CSV)", dft.to_csv(index=False), "truth_table.csv")

with tabs[2]:
    st.subheader("Canonical Forms")
    st.markdown(f"**Σm({', '.join(map(str, ones))})**" + (f"  +  d({', '.join(map(str, dcs))})" if dcs else ""))
    st.markdown("**Canonical SOP (sum of minterms)**")
    st.code("F = " + canon_sop, language=None)
    st.markdown(f"**ΠM({', '.join(map(str, zeros))})**")
    st.markdown("**Canonical POS (product of maxterms)**")
    st.code("F = " + canon_pos, language=None)
    st.markdown("**Minterm binary codes**")
    st.write(pd.DataFrame({"Minterm": [f"m{m}" for m in ones],
                           "Binary": [format(m, f"0{n}b") for m in ones],
                           "Term": [term_sop(format(m, f"0{n}b"), names) for m in ones]}))

with tabs[3]:
    st.subheader("Quine–McCluskey Method")
    st.markdown("#### Step 1 – Grouping by number of 1s and combining")
    for ci, col in enumerate(columns):
        st.markdown(f"**Column {ci+1}**")
        rows = []
        for k, p in sorted(col.items(), key=lambda kv: (kv[1].count("1"), kv[1])):
            used = any(k not in [kk for kk, _ in primes] for _ in [0])
            rows.append({"Ones": p.count("1"),
                         "Minterms": ",".join(map(str, k)),
                         "Pattern": p,
                         "Prime?": "✔ prime" if (k, p) in primes else ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown("#### Step 2 – Prime implicants")
    pi_rows = [{"PI": f"P{i+1}", "Minterms": ",".join(map(str, p[0])), "Pattern": p[1],
                "Term": term_sop(p[1], names)} for i, p in enumerate(primes)]
    st.dataframe(pd.DataFrame(pi_rows), hide_index=True)

    st.markdown("#### Step 3 – Prime implicant chart")
    if ones and primes:
        chart = {}
        for i, p in enumerate(primes):
            chart[f"P{i+1} ({term_sop(p[1], names)})"] = ["✖" if m in p[0] else "" for m in ones]
        chart_df = pd.DataFrame(chart, index=[f"m{m}" for m in ones]).T
        st.dataframe(chart_df)
    st.markdown("#### Step 4 – Essential prime implicants & selection")
    id_of = {p: f"P{i+1}" for i, p in enumerate(primes)}
    st.markdown("**Essential:** " + (", ".join(f"`{id_of[p]} = {term_sop(p[1], names)}`" for p in essential) or "none"))
    st.markdown("**Extra (to cover remaining minterms):** " + (", ".join(f"`{id_of[p]} = {term_sop(p[1], names)}`" for p in extra) or "none needed"))
    st.success(f"Minimal SOP:  F = {simp_sop}")

with tabs[4]:
    st.subheader("Karnaugh Map")
    if n > 4:
        st.info("K-map is drawn for 2–4 variables. Use the Quine–McCluskey tab for larger functions.")
    elif n < 2:
        st.info("Need at least 2 variables.")
    else:
        st.markdown(kmap_html(n, names, set(ones), set(dcs), cover), unsafe_allow_html=True)
        st.write("")
        st.markdown("**Groups selected**")
        for i, p in enumerate(cover):
            st.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;background:{COLORS[i % len(COLORS)]};border-radius:3px;margin-right:8px'></span>"
                f"Group {i+1}: minterms {','.join(map(str, p[0]))} → <code>{term_sop(p[1], names)}</code>",
                unsafe_allow_html=True)

with tabs[5]:
    st.subheader("Logic Circuit (Minimal SOP)")
    if not cover or any(set(p[1]) == {"-"} for p in cover):
        st.info(f"Function is constant: F = {simp_sop}")
    else:
        st.graphviz_chart(circuit_dot(names, cover))
        gates_and = sum(1 for p in cover if lit_count(p[1]) > 1)
        gates_not = len({v for p in cover for ch, v in zip(p[1], names) if ch == "0"})
        gates_or = 1 if len(cover) > 1 else 0
        g = st.columns(4)
        g[0].metric("AND gates", gates_and)
        g[1].metric("OR gates", gates_or)
        g[2].metric("NOT gates", gates_not)
        g[3].metric("Total gates", gates_and + gates_or + gates_not)

with tabs[6]:
    st.subheader("Verification & Report")
    st.markdown(f"- Original → **{lit_orig}** literals\n- Canonical SOP → **{lit_canon}** literals\n- Minimal SOP → **{lit_simp}** literals  (**{red}% reduction** vs original)")
    st.markdown(f"- SymPy cross-check (DNF): `{sym_str}`")
    st.markdown(f"- Truth-table equivalence: {'✅ PASS' if verified else '❌ FAIL'}")

    rep = io.StringIO()
    rep.write("BOOLEAN EXPRESSION SIMPLIFICATION REPORT\n")
    rep.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n" + "=" * 50 + "\n")
    rep.write(f"Input: {text if mode == 'Boolean expression' else 'Minterms ' + str(ones)}\n")
    rep.write(f"Variables: {', '.join(names)}\n")
    rep.write(f"Minterms: {ones}\nDon't cares: {dcs}\nMaxterms: {zeros}\n\n")
    rep.write(f"Canonical SOP: F = {canon_sop}\nCanonical POS: F = {canon_pos}\n\n")
    rep.write("Prime implicants:\n")
    for i, p in enumerate(primes):
        rep.write(f"  P{i+1}: m({','.join(map(str, p[0]))})  {p[1]}  ->  {term_sop(p[1], names)}\n")
    rep.write("\nEssential: " + ", ".join(term_sop(p[1], names) for p in essential) + "\n")
    rep.write(f"\nMinimal SOP: F = {simp_sop}\nMinimal POS: F = {simp_pos}\n")
    rep.write(f"Literals: {lit_orig} -> {lit_simp} ({red}% reduction)\n")
    rep.write(f"Verified: {'YES' if verified else 'NO'}\n")
    st.download_button("⬇️ Download full report (TXT)", rep.getvalue(), "boolean_report.txt")
    st.text_area("Report preview", rep.getvalue(), height=300)
