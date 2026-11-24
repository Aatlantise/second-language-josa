"""RQ2: which grammatical/phonological features predict postposition omission and choice,
and are they the same in L1 and L2 Korean?  Mixed-effects logistic regressions
(random intercept per writer; Laplace approximation, see glmm.py).

Outcomes
  omission    : argument root (subject/object) carries no postposition
  topic       : among marked subjects with 은/는 or 이/가, the root is topic-marked (은/는)
  allomorph   : (learners only) a postposition with an allomorph pair has the wrong allomorph
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import glmm
from common import RESULTS, WORK
from rq1 import bh

BLOCKS_ARG = {
    "relation": ["role_obj"],
    "root_pos": ["pos_PRON", "pos_OTHER"],
    "head_distance": ["head_dist"],
    "canonical_order": ["order_SOV", "order_OSV"],
    "clause_type": ["clause_sub"],
    "sentence_length": ["sent_len"],
}
BLOCKS_TOPIC = {k: v for k, v in BLOCKS_ARG.items() if k != "relation"}
# PROPN/NUM subjects are (quasi-)separated in KSL for topic vs nominative marking: keep only the pronoun contrast
BLOCKS_TOPIC["root_pos"] = ["pos_PRON"]
BLOCKS_ALLO = {"root_coda": ["coda_C", "coda_L"], "root_frequency": ["root_freq"]}


def arg_features(a):
    X = pd.DataFrame(index=a.index)
    X["role_obj"] = (a.role == "obj").astype(float)
    X["pos_PRON"] = (a.upos == "PRON").astype(float)
    X["pos_OTHER"] = a.upos.isin(["PROPN", "NUM"]).astype(float)
    X["head_dist"] = np.log1p(a.head_dist)
    X["order_SOV"] = (a.order == "SOV").astype(float)
    X["order_OSV"] = (a.order == "OSV").astype(float)
    X["clause_sub"] = (a.clause == "sub").astype(float)
    X["sent_len"] = np.log(a.sent_len)
    return X


def standardise(X, cols):
    X = X.copy()
    for c in cols:
        X[c] = (X[c] - X[c].mean()) / X[c].std()
    return X


NUM_COLS = ["head_dist", "sent_len", "root_freq"]


def per_corpus(df, y, blocks, l1, label, out_rows, sigma_rows):
    feats = [c for b in blocks.values() for c in b]
    X = standardise(df[feats], [c for c in NUM_COLS if c in feats])
    names = ["(Intercept)"] + feats
    mats = [np.ones(len(df)), *[X[c].values for c in feats]]
    if l1:
        for g in ("Czech", "English", "Japanese"):
            mats.append((df.group == g).astype(float).values)
            names.append(f"L1_{g} (ref Chinese)")
    M = np.column_stack(mats)
    full = glmm.fit(M, df[y].values, df.writer.values, names)
    for r in glmm.coef_table(full):
        out_rows.append(dict(model=label, **r))
    sigma_rows.append(dict(model=label, n=full["n"], writers=full["n_groups"], events=int(df[y].sum()),
                           sigma_writer=full["sigma"], converged=full["success"], loglik=full["ll"]))
    if l1:      # LRT for L1 overall, controlling for the features
        keep = [i for i, n in enumerate(names) if not n.startswith("L1_")]
        red = glmm.fit(M[:, keep], df[y].values, df.writer.values, [names[i] for i in keep], se=False)
        stat, d, p = glmm.lrt(full, red)
        sigma_rows[-1].update(L1_LRT_chi2=stat, L1_LRT_df=d, L1_LRT_p=p)
    return full


def pooled(df, y, blocks, label, coef_rows, lrt_rows, sigma_rows):
    """Pooled L1/L2 model: L2 indicator, each feature, and L2 x feature interactions."""
    feats = [c for b in blocks.values() for c in b]
    X = standardise(df[feats], [c for c in NUM_COLS if c in feats])
    L2 = (df.corpus == "KSL").astype(float).values
    names = ["(Intercept)", "L2"] + feats + [f + ":L2" for f in feats]
    M = np.column_stack([np.ones(len(df)), L2, *[X[c].values for c in feats], *[X[c].values * L2 for c in feats]])
    full = glmm.fit(M, df[y].values, df.writer.values, names)
    cov = full["cov"]
    z = norm.ppf(.975)
    nf = len(feats)
    for k, f in enumerate(feats):
        i_main, i_int = 2 + k, 2 + nf + k
        b1 = full["beta"][i_main]
        bi = full["beta"][i_int]
        v1 = cov[i_main, i_main]
        vi = cov[i_int, i_int]
        c1i = cov[i_main, i_int]
        s2 = np.sqrt(max(v1 + vi + 2 * c1i, 0))
        coef_rows.append(dict(
            model=label, feature=f,
            OR_L1=np.exp(b1), OR_L1_lo=np.exp(b1 - z * np.sqrt(v1)), OR_L1_hi=np.exp(b1 + z * np.sqrt(v1)),
            OR_L2=np.exp(b1 + bi), OR_L2_lo=np.exp(b1 + bi - z * s2), OR_L2_hi=np.exp(b1 + bi + z * s2),
            OR_interaction_L2_vs_L1=np.exp(bi), OR_int_lo=np.exp(bi - z * np.sqrt(vi)),
            OR_int_hi=np.exp(bi + z * np.sqrt(vi)),
            p_interaction_wald=2 * norm.sf(abs(bi / np.sqrt(vi))),
            L2_effect_weaker=bool(abs(b1 + bi) < abs(b1))))
    sigma_rows.append(dict(model=label, n=full["n"], writers=full["n_groups"], events=int(df[y].sum()),
                           sigma_writer=full["sigma"], converged=full["success"], loglik=full["ll"],
                           L2_main_OR=np.exp(full["beta"][1])))
    rows = []
    for bname, cols in blocks.items():
        drop = {f + ":L2" for f in cols}
        keep = [i for i, n in enumerate(names) if n not in drop]
        red = glmm.fit(M[:, keep], df[y].values, df.writer.values, [names[i] for i in keep], se=False)
        stat, d, p = glmm.lrt(full, red)
        rows.append(dict(model=label, block=bname, chi2=stat, df=d, p=p))
    q = bh([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q_BH"] = qq
    lrt_rows.extend(rows)


def main():
    tok = pd.read_pickle(WORK / "tokens_flagged.pkl")
    args = pd.read_pickle(WORK / "args.pkl")
    args = args[args.order != "other"].copy()
    ksl, kaist = args[args.corpus == "KSL"], args[args.corpus == "KAIST"]
    coef_rows, sigma_rows, pooled_coef, lrt_rows = [], [], [], []

    # ---------------- omission
    om = args.copy()
    om["omit"] = 1 - om.marked
    feats = arg_features(om)
    om = pd.concat([om.drop(columns=feats.columns, errors='ignore'), feats], axis=1)
    for name, d, l1 in (("omission | KSL", om[om.corpus == "KSL"], True), ("omission | KAIST", om[om.corpus == "KAIST"], False)):
        per_corpus(d, "omit", BLOCKS_ARG, l1, name, coef_rows, sigma_rows)
    pooled(om, "omit", BLOCKS_ARG, "omission", pooled_coef, lrt_rows, sigma_rows)

    # ---------------- topic vs nominative (marked subjects)
    tp = args[(args.role == "subj") & args.category.isin(["TOP", "NOM"])].copy()
    tp["topic"] = (tp.category == "TOP").astype(int)
    f2 = arg_features(tp)
    tp = pd.concat([tp.drop(columns=f2.columns, errors='ignore'), f2], axis=1)
    for name, d, l1 in (("topic | KSL", tp[tp.corpus == "KSL"], True), ("topic | KAIST", tp[tp.corpus == "KAIST"], False)):
        per_corpus(d, "topic", BLOCKS_TOPIC, l1, name, coef_rows, sigma_rows)
    pooled(tp, "topic", BLOCKS_TOPIC, "topic", pooled_coef, lrt_rows, sigma_rows)

    # ---------------- allomorph errors (learners only)
    freq = tok[(tok.corpus == "KAIST") & (tok.stem != "")].groupby("stem").size()
    al = tok[(tok.corpus == "KSL") & (tok.allomorph_eval > 0) & tok.stem_coda.notna() & (tok.stem != "")].copy()
    al["err"] = (al.allomorph_err > 0).astype(int)
    al["coda_C"] = (al.stem_coda == "C").astype(float)
    al["coda_L"] = (al.stem_coda == "L").astype(float)
    al["root_freq"] = np.log1p(al.stem.map(freq).fillna(0))
    per_corpus(al, "err", BLOCKS_ALLO, True, "allomorph_error | KSL", coef_rows, sigma_rows)

    pd.DataFrame(coef_rows).to_csv(RESULTS / "rq2_models_per_corpus.csv", index=False)
    pd.DataFrame(sigma_rows).to_csv(RESULTS / "rq2_model_summaries.csv", index=False)
    pd.DataFrame(pooled_coef).to_csv(RESULTS / "rq2_pooled_L1_vs_L2.csv", index=False)
    pd.DataFrame(lrt_rows).to_csv(RESULTS / "rq2_pooled_interaction_LRT.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(pd.DataFrame(sigma_rows).round(3).to_string())
    print(pd.DataFrame(coef_rows).round(3).to_string())
    print(pd.DataFrame(pooled_coef).round(3).to_string())
    print(pd.DataFrame(lrt_rows).round(4).to_string())


if __name__ == "__main__":
    main()
