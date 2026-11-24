"""Manual-validation workflow for the postposition extraction (paper: Validation).

  python validation.py make     -> results/validation/sheet_annotator1.csv (+ annotator2 subset)
  python validation.py score    -> precision / recall / accuracy with 95% Wilson intervals,
                                   Cohen's kappa between annotators (needs the filled sheets)

The sample is drawn from the *test* splits of KSL (50 sentences per L1) and KAIST (200 sentences).
Annotators fill the `gold_*` columns ONLY where they DISAGREE with the extraction (blank = agree):
  gold_pp_lemmas   correct postposition sequence for the token, e.g. 에+은/는 ; NONE if the token
                   has no postposition (extraction false positive)
  gold_arg_role    subj / obj / NONE : correct argument status of the token (if different)
  gold_omission    y / n : whether this argument root really lacks a required-looking postposition
                   (only for argument roots; leave blank if the extraction is right)
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import RESULTS, SEED, WORK

VAL = RESULTS / "validation"


def wilson(k, n, level=0.95):
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(0.5 + level / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def make():
    VAL.mkdir(exist_ok=True)
    tok = pd.read_pickle(WORK / "tokens_flagged.pkl")
    args = pd.read_pickle(WORK / "args.pkl")
    rng = np.random.default_rng(SEED)
    t = tok[tok.split == "test"]
    chosen = []
    for g, n in (("Chinese", 50), ("Czech", 50), ("English", 50), ("Japanese", 50), ("L1 Korean", 200)):
        sn = t[t.group == g].sno.unique()
        chosen += list(rng.choice(sn, size=min(n, len(sn)), replace=False))
    s = t[t.sno.isin(chosen)].copy()
    a = args.set_index(["sno", "i"])
    s["extracted_arg_role"] = [a.role.get((sn, i), "") for sn, i in zip(s.sno, s.i)]
    s["extracted_omitted"] = [("y" if a.marked.get((sn, i), 1) == 0 else "n") if (sn, i) in a.index else ""
                              for sn, i in zip(s.sno, s.i)]
    cols = ["group", "writer", "sent_id", "i", "form", "lemma", "xpos", "upos", "deprel", "head",
            "n_pp", "pp_lemmas", "pp_classes", "extracted_arg_role", "extracted_omitted"]
    out = s[cols].rename(columns={"n_pp": "extracted_n_pp", "pp_lemmas": "extracted_pp_lemmas",
                                  "pp_classes": "extracted_classes"})
    for c in ("gold_pp_lemmas", "gold_arg_role", "gold_omission", "notes"):
        out[c] = ""
    out["sno"] = s.sno.values
    out = out.sort_values(["group", "sno", "i"])
    out.drop(columns="sno").to_csv(VAL / "sheet_annotator1.csv", index=False, encoding="utf-8-sig")
    # second annotator: 20% of the sentences of each group
    sub = []
    for g, d in out.groupby("group"):
        sn = d.sno.unique()
        sub += list(rng.choice(sn, size=int(round(.2 * len(sn))), replace=False))
    out[out.sno.isin(sub)].drop(columns="sno").to_csv(VAL / "sheet_annotator2.csv", index=False, encoding="utf-8-sig")
    print(f"sentences: {len(chosen)}, tokens: {len(out)}; second-annotator sentences: {len(sub)}")


def resolved(df):
    """Return gold postposition string (NONE if none) per token after applying disagreements."""
    ext = df.extracted_pp_lemmas.fillna("").replace("", "NONE")
    gold = df.gold_pp_lemmas.fillna("").astype(str).str.strip()
    return np.where(gold == "", ext, gold)


def score():
    a1 = pd.read_csv(VAL / "sheet_annotator1.csv", encoding="utf-8-sig", dtype=str).fillna("")
    a1["corpus"] = np.where(a1.group == "L1 Korean", "KAIST", "KSL")
    a1["gold"] = resolved(a1)
    a1["ext"] = a1.extracted_pp_lemmas.replace("", "NONE")
    a1["ext_has"] = a1.ext != "NONE"
    a1["gold_has"] = a1.gold != "NONE"
    rows = []
    for key, d in list(a1.groupby("group")) + list(a1.groupby("corpus")) + [("ALL", a1)]:
        tp = int((d.ext_has & d.gold_has).sum())
        fp = int((d.ext_has & ~d.gold_has).sum())
        fn = int((~d.ext_has & d.gold_has).sum())
        exact = int((d.ext == d.gold).sum())
        for name, k, n in (("precision", tp, tp + fp), ("recall", tp, tp + fn), ("token_accuracy", exact, len(d))):
            lo, hi = wilson(k, n)
            rows.append(dict(unit=key, metric=name, k=k, n=n, estimate=k / n if n else np.nan, ci_lo=lo, ci_hi=hi))
        ar = d[d.extracted_arg_role != ""]
        ok = int(((ar.gold_arg_role == "") | (ar.gold_arg_role == ar.extracted_arg_role)).sum())
        lo, hi = wilson(ok, len(ar))
        rows.append(dict(unit=key, metric="argument_root_precision", k=ok, n=len(ar),
                         estimate=ok / len(ar) if len(ar) else np.nan, ci_lo=lo, ci_hi=hi))
    res = pd.DataFrame(rows)
    res.to_csv(VAL / "validation_scores.csv", index=False)
    print(res.round(3).to_string())
    try:
        a2 = pd.read_csv(VAL / "sheet_annotator2.csv", encoding="utf-8-sig", dtype=str).fillna("")
        a2["gold"] = resolved(a2)
        m = a1[["sent_id", "i", "gold"]].merge(a2[["sent_id", "i", "gold"]], on=["sent_id", "i"], suffixes=("_1", "_2"))
        g1, g2 = m.gold_1, m.gold_2
        h1, h2 = (g1 != "NONE"), (g2 != "NONE")
        po = (h1 == h2).mean()
        pe = h1.mean() * h2.mean() + (1 - h1.mean()) * (1 - h2.mean())
        print("Cohen's kappa (postposition present/absent):", round((po - pe) / (1 - pe), 3), "n =", len(m))
    except FileNotFoundError:
        print("no second-annotator sheet found")


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
