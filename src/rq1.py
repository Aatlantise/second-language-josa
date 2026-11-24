"""RQ1: how does L2 postposition use differ from L1 Korean (distribution, collocation,
omission, choice, stacking, errors), overall and by L1?   -> results/rq1_*.csv, figures
"""
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (ALL_GROUPS, BASELINE, L2_GROUPS, N_BOOT, RESULTS, SEED, WORK)

NOMINAL = {"NOUN", "PRON", "PROPN", "NUM"}
SUBST_DEPRELS = {"nsubj", "obj"}   # KAIST labels nsubj/obj only on case-marked roots (see DECISIONS.md)
SUBST_THRESH = 0.01      # first postposition with <1% baseline share for the deprel => suspect
MIN_DOC_DEN = 30         # docs with fewer relevant tokens are ignored for the baseline range

INTERFACE = ["subj_omission", "obj_omission", "topic_choice"]
NARROW = ["acc_choice", "allomorph_error"]
# acc_choice is excluded from composites: KAIST labels obj only on 를-marked roots, so its baseline is 100% by construction
H1_CORE = ["subj_omission", "obj_omission", "topic_choice", "stacking_rate"]
ERR = ["allomorph_error", "substitution_proxy", "invalid_form"]
MEASURES = ["subj_omission", "obj_omission", "omission_canonical_order", "omission_object_first",
            "topic_choice", "acc_choice", "stacking_rate"] + ERR


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def load():
    tok = pd.read_pickle(WORK / "tokens.pkl")
    args = pd.read_pickle(WORK / "args.pkl")
    return tok, args


# ---------------------------------------------------------- error proxies
def add_error_flags(tok):
    """Leave-one-document-out baseline rates for the rule-based error proxies."""
    tok = tok.copy()
    base = tok.group == BASELINE
    # ---- substitution proxy: first postposition rare in the baseline for this relation
    sel = ((tok.pp_lemmas != "") & tok.upos.isin(NOMINAL) & tok.deprel.isin(SUBST_DEPRELS)
           & (tok.first_pp_class == "case") & ~tok.detached)
    tok["subst_eval"] = sel
    b = tok[sel & base]
    c_tot = b.groupby(["deprel", "first_pp"]).size()
    t_tot = b.groupby("deprel").size()
    c_doc = b.groupby(["writer", "deprel", "first_pp"]).size()
    t_doc = b.groupby(["writer", "deprel"]).size()
    flag = pd.Series(False, index=tok.index)
    s = tok[sel]
    key = list(zip(s.deprel, s.first_pp))
    num = np.array([c_tot.get(k, 0) for k in key], float)
    den = s.deprel.map(t_tot).to_numpy(float)
    isb = (s.group == BASELINE).to_numpy()
    if isb.any():
        sb = s[isb]
        num[isb] -= np.array([c_doc.get((w, d, f), 0) for w, d, f in zip(sb.writer, sb.deprel, sb.first_pp)])
        den[isb] -= np.array([t_doc.get((w, d), 0) for w, d in zip(sb.writer, sb.deprel)])
    flag.loc[s.index] = (num / den) < SUBST_THRESH
    tok["subst_flag"] = flag
    # ---- invalid form: stack unattested in the baseline (outside the token's own document)
    sel2 = (tok.pp_lemmas != "") & ~tok.detached
    tok["invalid_eval"] = sel2
    b2 = tok[sel2 & base]
    s_tot = b2.groupby("pp_lemmas").size()
    s_doc = b2.groupby(["writer", "pp_lemmas"]).size()
    s2 = tok[sel2]
    n_out = np.array([s_tot.get(k, 0) for k in s2.pp_lemmas], float)
    isb2 = (s2.group == BASELINE).to_numpy()
    sb2 = s2[isb2]
    n_out[isb2] -= np.array([s_doc.get((w, k), 0) for w, k in zip(sb2.writer, sb2.pp_lemmas)])
    inv = pd.Series(False, index=tok.index)
    inv.loc[s2.index] = n_out <= 0
    tok["invalid_flag"] = inv
    return tok


def proxy_validation(tok):
    """Recall of the substitution proxy against the treebank's own CaseError comments."""
    ks = tok[(tok.corpus == "KSL") & tok.comment.str.startswith("CaseError")]
    ev = ks[ks.subst_eval]
    lines = [f"CaseError-annotated KSL tokens: {len(ks)}; of which evaluable by the proxy: {len(ev)}; "
             f"flagged by the proxy: {int(ev.subst_flag.sum())} "
             f"(recall {ev.subst_flag.mean():.2f})"]
    ks_all = tok[(tok.corpus == "KSL") & tok.subst_eval]
    lines.append(f"Proxy flags {int(ks_all.subst_flag.sum())} of {len(ks_all)} evaluable KSL tokens "
                 f"({ks_all.subst_flag.mean():.3f}); CaseError comments cover only a handful of tokens, "
                 "so precision cannot be estimated from them (see validation sheets).")
    (RESULTS / "rq1_proxy_check.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


# ------------------------------------------------------- writer-level counts
def writer_table(tok, args):
    """dict measure -> DataFrame(index=writer, columns=num, den, group)."""
    writers = tok.drop_duplicates("writer").set_index("writer")["group"]
    out = {}

    def agg(df, num, den, name):
        g = pd.DataFrame({"num": num, "den": den, "writer": df.writer.values}).groupby("writer").sum()
        g = g.reindex(writers.index, fill_value=0)
        g["group"] = writers
        out[name] = g

    for role, name in (("subj", "subj_omission"), ("obj", "obj_omission")):
        a = args[args.role == role]
        agg(a, (1 - a.marked).values, np.ones(len(a)), name)
    a = args[args.order == "SOV"]
    agg(a, (1 - a.marked).values, np.ones(len(a)), "omission_canonical_order")
    a = args[args.order == "OSV"]
    agg(a, (1 - a.marked).values, np.ones(len(a)), "omission_object_first")
    a = args[(args.role == "subj") & args.category.isin(["TOP", "NOM"])]
    agg(a, (a.category == "TOP").astype(int).values, np.ones(len(a)), "topic_choice")
    a = args[(args.role == "obj") & (args.marked == 1)]
    agg(a, (a.category == "ACC").astype(int).values, np.ones(len(a)), "acc_choice")
    t = tok[tok.n_pp > 0]
    agg(t, (t.n_pp >= 2).astype(int).values, np.ones(len(t)), "stacking_rate")
    agg(tok, tok.allomorph_err.values, tok.allomorph_eval.values, "allomorph_error")
    t = tok[tok.subst_eval]
    agg(t, t.subst_flag.astype(int).values, np.ones(len(t)), "substitution_proxy")
    t = tok[tok.invalid_eval]
    agg(t, t.invalid_flag.astype(int).values, np.ones(len(t)), "invalid_form")
    return out


def boot_indices(sizes, rng, B):
    return {g: rng.integers(0, n, size=(B, n)) for g, n in sizes.items()}


def lor(x, n, xb, nb):
    return np.log(((x + .5) / (n - x + .5)) / ((xb + .5) / (nb - xb + .5)))


def pct(a, lo=2.5, hi=97.5, axis=0):
    return np.percentile(a, [lo, hi], axis=axis)


def run_measures(wt, rng):
    """Point estimates, bootstrap CIs, LORs, H1 pairwise, H3, baseline ranges."""
    wr = {g: {m: wt[m][wt[m].group == g] for m in wt} for g in ALL_GROUPS}
    sizes = {g: len(wr[g][MEASURES[0]]) for g in ALL_GROUPS}
    idx = boot_indices(sizes, rng, N_BOOT)
    rows, lor_boot, lor_pt = [], {}, {}
    h_boot, h_pt = {}, {}
    phi = lambda x, n: 2 * np.arcsin(np.sqrt(np.clip(x / n, 0, 1)))
    for m in MEASURES:
        xb_pt, nb_pt = wr[BASELINE][m].num.sum(), wr[BASELINE][m].den.sum()
        bnum, bden = wr[BASELINE][m].num.values, wr[BASELINE][m].den.values
        Xb, Nb = bnum[idx[BASELINE]].sum(1), bden[idx[BASELINE]].sum(1)
        docp = wr[BASELINE][m]
        docp = (docp.num / docp.den)[docp.den >= MIN_DOC_DEN]
        rng_lo, rng_hi = (docp.quantile(.025), docp.quantile(.975)) if len(docp) > 5 else (np.nan, np.nan)
        for g in ALL_GROUPS:
            num, den = wr[g][m].num.values, wr[g][m].den.values
            X, N = num[idx[g]].sum(1), den[idx[g]].sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                p_b = X / N
            est = num.sum() / den.sum() if den.sum() else np.nan
            lo, hi = np.nanpercentile(p_b, [2.5, 97.5])
            r = dict(measure=m, group=g, num=num.sum(), den=den.sum(), estimate=est, ci_lo=lo, ci_hi=hi,
                     base_doc_p2_5=rng_lo, base_doc_p97_5=rng_hi, base_doc_min=docp.min(), base_doc_max=docp.max())
            if g != BASELINE:
                L = lor(X, N, Xb, Nb)
                pt = lor(num.sum(), den.sum(), xb_pt, nb_pt)
                lor_boot[(m, g)], lor_pt[(m, g)] = L, pt
                with np.errstate(divide="ignore", invalid="ignore"):
                    h_boot[(m, g)] = np.abs(phi(X, N) - phi(Xb, Nb))
                h_pt[(m, g)] = abs(phi(num.sum(), den.sum()) - phi(xb_pt, nb_pt))
                r.update(lor=pt, lor_lo=np.nanpercentile(L, 2.5), lor_hi=np.nanpercentile(L, 97.5),
                         outside_baseline_doc_range=bool(est < rng_lo or est > rng_hi))
            rows.append(r)
    meas = pd.DataFrame(rows)

    # ---- H1: is L1-Japanese closer (smaller |LOR|) than the others?
    comps = {
        "core(omission,choice,stacking)": H1_CORE,
        "all_measures_incl_errors": H1_CORE + ERR,
    }
    ab_boot = {(m, g): np.abs(lor_boot[(m, g)]) for (m, g) in lor_boot}
    ab_pt = {k: abs(v) for k, v in lor_pt.items()}
    h1 = []
    units = [(m, [m]) for m in MEASURES] + [(k, v) for k, v in comps.items()]
    for name, ms in units:
        B = {g: np.mean([ab_boot[(m, g)] for m in ms], axis=0) for g in L2_GROUPS}
        P = {g: np.mean([ab_pt[(m, g)] for m in ms]) for g in L2_GROUPS}
        for a, b in itertools.combinations(L2_GROUPS, 2):
            d = B[a] - B[b]
            h1.append(dict(unit=name, group_a=a, group_b=b, abs_lor_a=P[a], abs_lor_b=P[b],
                           diff_a_minus_b=P[a] - P[b], lo=np.nanpercentile(d, 2.5),
                           hi=np.nanpercentile(d, 97.5),
                           excludes_zero=bool(np.nanpercentile(d, 2.5) > 0 or np.nanpercentile(d, 97.5) < 0)))
    h1 = pd.DataFrame(h1)

    # ---- H3: interface-conditioned vs narrow-syntax deviations, within each group
    h3 = []
    for g in L2_GROUPS:
        bi = np.mean([ab_boot[(m, g)] for m in INTERFACE], axis=0)
        bn = np.mean([ab_boot[(m, g)] for m in NARROW], axis=0)
        pi = np.mean([ab_pt[(m, g)] for m in INTERFACE])
        pn = np.mean([ab_pt[(m, g)] for m in NARROW])
        d = bi - bn
        h3.append(dict(group=g, mean_abs_lor_interface=pi, mean_abs_lor_narrow=pn, diff=pi - pn,
                       lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                       supports_H3=bool(np.nanpercentile(d, 2.5) > 0)))
        # sensitivity variants: narrow = allomorph error only / accusative choice only
        bn2 = ab_boot[("allomorph_error", g)]
        d2 = bi - bn2
        h3[-1].update(diff_vs_allomorph_only=pi - ab_pt[("allomorph_error", g)],
                      lo_allomorph_only=np.nanpercentile(d2, 2.5), hi_allomorph_only=np.nanpercentile(d2, 97.5))
        bn3 = ab_boot[("acc_choice", g)]
        d3 = bi - bn3
        h3[-1].update(diff_vs_acc_only=pi - ab_pt[("acc_choice", g)], lo_acc_only=np.nanpercentile(d3, 2.5),
                      hi_acc_only=np.nanpercentile(d3, 97.5))
    # ---- H3 with Cohen's h (arcsine effect size): bounded, finite for ~0 baselines
    h3h = []
    for g in L2_GROUPS:
        for label, narrow in (("acc+allomorph (as specified)", NARROW), ("allomorph only", ["allomorph_error"]),
                              ("acc only (convention artifact)", ["acc_choice"])):
            bi = np.mean([h_boot[(m, g)] for m in INTERFACE], axis=0)
            bn = np.mean([h_boot[(m, g)] for m in narrow], axis=0)
            pi = np.mean([h_pt[(m, g)] for m in INTERFACE])
            pn = np.mean([h_pt[(m, g)] for m in narrow])
            d = bi - bn
            h3h.append(dict(group=g, narrow_set=label, mean_h_interface=pi, mean_h_narrow=pn, diff=pi - pn,
                            lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                            supports_H3=bool(np.nanpercentile(d, 2.5) > 0)))
    pd.DataFrame(h3h).to_csv(RESULTS / "rq1_h3_cohens_h.csv", index=False)
    return meas, h1, pd.DataFrame(h3), idx


# ------------------------------------------------------------- distribution
def g2(a, n1, b, n2):
    """Dunning log-likelihood ratio for count a of n1 vs count b of n2 (vectorised)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    e1 = n1 * (a + b) / (n1 + n2)
    e2 = n2 * (a + b) / (n1 + n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(a > 0, a * np.log(a / e1), 0.0)
        t2 = np.where(b > 0, b * np.log(b / e2), 0.0)
    return 2 * (t1 + t2)


def distribution(tok, idx, base_dp=None):
    pp = tok[tok.pp_lemmas != ""]
    writers = tok.drop_duplicates("writer").set_index("writer")["group"]
    tokens_w = tok.groupby("writer").size().reindex(writers.index)
    # writer x lemma matrix
    ex = pp.assign(l=pp.pp_lemmas.str.split("+")).explode("l")
    mat = ex.groupby(["writer", "l"]).size().unstack(fill_value=0).reindex(writers.index, fill_value=0)
    cls = pp.assign(c=pp.pp_classes.str.split("+")).explode("c").groupby(["writer", "c"]).size() \
        .unstack(fill_value=0).reindex(writers.index, fill_value=0)
    res, cls_rows = [], []
    G = {g: np.where(writers.values == g)[0] for g in ALL_GROUPS}
    M, T = mat.values.astype(float), tokens_w.values.astype(float)
    bi = idx[BASELINE]
    Nb_all = T[G[BASELINE]]
    Mb = M[G[BASELINE]]
    Nbb = Nb_all[bi].sum(1)
    Nb_pt = Nb_all.sum()
    out_log2 = {}
    for g in L2_GROUPS:
        Mg, Tg = M[G[g]], T[G[g]]
        gi = idx[g]
        a, b = Mg.sum(0), Mb.sum(0)
        Ng, Nb = Tg.sum(), Nb_pt
        keep = (b * Ng / Nb >= 5) | (a >= 5)
        l2 = np.log2(((a + .5) / Ng) / ((b + .5) / Nb))
        A = np.stack([Mg[gi[k]].sum(0) for k in range(N_BOOT)])      # B x lemmas
        Bm = np.stack([Mb[bi[k]].sum(0) for k in range(N_BOOT)])
        NgB = Tg[gi].sum(1)[:, None]
        L2B = np.log2(((A + .5) / NgB) / ((Bm + .5) / Nbb[:, None]))
        lo, hi = np.percentile(L2B, [2.5, 97.5], axis=0)
        pv = 2 * np.minimum((L2B >= 0).mean(0), (L2B <= 0).mean(0))
        pv = np.maximum(pv, 1 / (N_BOOT + 1))
        df = pd.DataFrame(dict(group=g, lemma=mat.columns, learner_n=a, baseline_n=b,
                               learner_per1000=1000 * a / Ng, baseline_per1000=1000 * b / Nb,
                               log2_ratio=l2, ci_lo=lo, ci_hi=hi, G2=g2(a, Ng, b, Nb), boot_p=pv))[keep]
        # KAIST tags the copula+connective 이고 as a conjunctive postposition, KSL tags it VCP+EC
        df["suspected_tagging_artifact"] = df.lemma.isin(["이고/고"])
        df["q_BH"] = bh(df.boot_p.values)
        df["sig_BH"] = (df.q_BH < .05) & ((df.ci_lo > 0) | (df.ci_hi < 0))
        res.append(df.sort_values("G2", ascending=False))
        out_log2[g] = pd.Series(l2, index=mat.columns)[keep]
        # classes
        for c in cls.columns:
            ca = cls.values[G[g]][:, list(cls.columns).index(c)]
            cb = cls.values[G[BASELINE]][:, list(cls.columns).index(c)]
            Xg = ca[gi].sum(1) / Tg[gi].sum(1) * 1000
            Xb = cb[bi].sum(1) / Nbb * 1000
            cls_rows.append(dict(group=g, cls=c, per1000=1000 * ca.sum() / Ng, baseline_per1000=1000 * cb.sum() / Nb,
                                 diff_lo=np.percentile(Xg - Xb, 2.5), diff_hi=np.percentile(Xg - Xb, 97.5)))
    dist = pd.concat(res)
    clsdf = pd.DataFrame(cls_rows)

    # ---- are deviations shared across L1 groups?  Spearman between groups' log2 ratios
    common = set.intersection(*[set(s.index) for s in out_log2.values()])
    common = sorted(common)
    sp = []
    for a, b in itertools.combinations(L2_GROUPS, 2):
        rho = spearmanr(out_log2[a][common], out_log2[b][common])[0]
        sp.append(dict(group_a=a, group_b=b, n_lemmas=len(common), spearman=rho))
    sp = pd.DataFrame(sp)

    # ---- usage-based: |log2 ratio| vs baseline frequency and delta-P
    nom = tok[(tok.group == BASELINE) & tok.upos.isin(NOMINAL)]
    exn = nom.assign(l=nom.pp_lemmas.str.split("+")).explode("l")
    dp = {}
    ntot = len(nom)
    for l in common:
        has = nom.pp_lemmas.str.split("+").map(lambda x: l in x)
        if has.sum() < 5:
            continue
        d_star = nom[has].deprel.value_counts().idxmax()
        p1 = (nom[has].deprel == d_star).mean()
        p0 = (nom[~has].deprel == d_star).mean()
        dp[l] = p1 - p0
    dp = pd.Series(dp)
    freq = pd.Series(Mb.sum(0), index=mat.columns)
    corr = []
    for g in L2_GROUPS:
        lem = [l for l in common if l in dp.index]
        ab = out_log2[g][lem].abs()
        corr.append(dict(group=g, n=len(lem),
                         rho_absLR_vs_log_baseline_freq=spearmanr(ab, np.log(freq[lem] + 1))[0],
                         rho_absLR_vs_deltaP=spearmanr(ab, dp[lem])[0]))
    return dist, clsdf, sp, pd.DataFrame(corr)


# ---------------------------------------------------- collocations & stacking
def g2_pair(n_pair, n_stem, n_stack, T):
    a = n_pair
    b = n_stem - a
    c = n_stack - a
    d = T - n_stem - n_stack + a
    obs = np.stack([a, b, c, d], 1).astype(float)
    row = np.stack([a + b, a + b, c + d, c + d], 1)
    col = np.stack([a + c, b + d, a + c, b + d], 1)
    exp = row * col / T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(obs > 0, obs * np.log(obs / exp), 0.0)
    return 2 * t.sum(1)


def colloc_stats(pairs, stacks):
    """pairs: list of (stem, stack) for pp tokens; stacks: list of stack strings (all pp tokens)."""
    T = len(pairs)
    if T == 0:
        return dict(pair_types=0, top10_mean_G2=np.nan, top10_token_share=np.nan,
                    stack_types=0, stack_top3_share=np.nan)
    pc = Counter(pairs)
    sc = Counter(p[0] for p in pairs)
    kc = Counter(p[1] for p in pairs)
    keep = [(k, v) for k, v in pc.items() if v >= 5]
    if keep:
        n_pair = np.array([v for _, v in keep], float)
        n_stem = np.array([sc[k[0]] for k, _ in keep], float)
        n_stack = np.array([kc[k[1]] for k, _ in keep], float)
        G = g2_pair(n_pair, n_stem, n_stack, float(T))
        top = np.sort(G)[::-1][:10]
        top_idx = np.argsort(G)[::-1][:10]
        share = n_pair[top_idx].sum() / T
    else:
        top, share = np.array([np.nan]), np.nan
    st = Counter(s for s in stacks if "+" in s)
    tot = sum(st.values())
    return dict(pair_types=len(keep), top10_mean_G2=float(np.nanmean(top)), top10_token_share=share,
                stack_types=len(st), stack_top3_share=(sum(v for _, v in st.most_common(3)) / tot) if tot else np.nan)


def sent_pool(tok, group):
    t = tok[(tok.group == group)]
    sizes = t.groupby("sno").size()
    pp = t[(t.pp_lemmas != "") & (t.stem != "") & ~t.detached]
    per = {s: list(zip(d.stem, d.pp_lemmas)) for s, d in pp.groupby("sno")}
    return sizes.index.values, sizes.values, per


def subsample(pool, budget, rng):
    sno, size, per = pool
    order = rng.permutation(len(sno))
    cum = np.cumsum(size[order])
    take = order[: np.searchsorted(cum, budget) + 1]
    pairs = [p for i in take for p in per.get(sno[i], [])]
    return pairs, [p[1] for p in pairs], int(size[take].sum())


def collocations(tok, rng, reps=200):
    pools = {g: sent_pool(tok, g) for g in ALL_GROUPS}
    ntok = tok.groupby("group").size()
    rows = []
    common = int(ntok[L2_GROUPS].min())
    # (1) each learner group in full vs baseline subsampled to the same size
    for g in L2_GROUPS:
        full_pairs = [p for l in pools[g][2].values() for p in l]
        obs = colloc_stats(full_pairs, [p[1] for p in full_pairs])
        rows.append(dict(comparison="group_full", group=g, tokens=int(ntok[g]), **obs))
        sims = [colloc_stats(*subsample(pools[BASELINE], ntok[g], rng)[:2]) for _ in range(reps)]
        sd = pd.DataFrame(sims)
        row = dict(comparison=f"baseline_subsampled_to_{g}", group=BASELINE, tokens=int(ntok[g]),
                   **sd.mean().to_dict())
        for c in sd.columns:
            row[c + "_lo"], row[c + "_hi"] = np.nanpercentile(sd[c], [2.5, 97.5])
        rows.append(row)
    # (2) all groups at a common size (the smallest learner group)
    for g in ALL_GROUPS:
        sims = pd.DataFrame([colloc_stats(*subsample(pools[g], common, rng)[:2]) for _ in range(reps)])
        row = dict(comparison=f"common_size_{common}", group=g, tokens=common, **sims.mean().to_dict())
        for c in sims.columns:
            row[c + "_lo"], row[c + "_hi"] = np.nanpercentile(sims[c], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def top_collocations(tok, n=15):
    out = []
    for g in ALL_GROUPS:
        t = tok[(tok.group == g) & (tok.pp_lemmas != "") & (tok.stem != "") & ~tok.detached]
        pairs = list(zip(t.stem, t.pp_lemmas))
        pc, sc, kc = Counter(pairs), Counter(p[0] for p in pairs), Counter(p[1] for p in pairs)
        keep = [(k, v) for k, v in pc.items() if v >= 5]
        if not keep:
            continue
        n_pair = np.array([v for _, v in keep], float)
        G = g2_pair(n_pair, np.array([sc[k[0]] for k, _ in keep], float),
                    np.array([kc[k[1]] for k, _ in keep], float), float(len(pairs)))
        for i in np.argsort(G)[::-1][:n]:
            out.append(dict(group=g, stem=keep[i][0][0], postposition=keep[i][0][1], n=keep[i][1], G2=G[i]))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures
def figures(meas, dist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    cols = {"Chinese": "#c44e52", "Czech": "#8172b2", "English": "#4c72b0", "Japanese": "#dd8452", BASELINE: "#222222"}
    ms = [m for m in MEASURES if m not in ("omission_object_first",)]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5))
    for ax, m in zip(axes.ravel(), ms):
        d = meas[meas.measure == m].set_index("group").loc[ALL_GROUPS]
        y = np.arange(len(d))
        ax.errorbar(d.estimate, y, xerr=[d.estimate - d.ci_lo, d.ci_hi - d.estimate], fmt="none",
                    ecolor="#888", capsize=2)
        ax.scatter(d.estimate, y, c=[cols[g] for g in d.index], zorder=3)
        lo, hi = d.base_doc_p2_5.iloc[0], d.base_doc_p97_5.iloc[0]
        if np.isfinite(lo):
            ax.axvspan(lo, hi, color="#ddd", alpha=.6, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(d.index if m == ms[0] or m == ms[5] else [""] * len(d))
        ax.set_title(m.replace("_", " "), fontsize=9)
        ax.invert_yaxis()
    for ax in axes.ravel()[len(ms):]:
        ax.axis("off")
    fig.suptitle("RQ1 measures by L1 group (95% bootstrap CI; grey band = middle 95% of KAIST documents)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_rq1_measures.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    show = [m for m in ms]
    w = .2
    for k, g in enumerate(L2_GROUPS):
        d = meas[(meas.group == g)].set_index("measure").loc[show]
        ax.bar(np.arange(len(show)) + k * w, d.lor.abs(), w, color=cols[g], label=g)
    ax.set_xticks(np.arange(len(show)) + 1.5 * w)
    ax.set_xticklabels([m.replace("_", "\n") for m in show], fontsize=7)
    ax.set_ylabel("|log odds ratio| vs L1 Korean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_rq1_abs_lor.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax, g in zip(axes, L2_GROUPS):
        d = dist[dist.group == g].sort_values("G2", ascending=False).head(15).iloc[::-1]
        ax.errorbar(d.log2_ratio, np.arange(len(d)), xerr=[d.log2_ratio - d.ci_lo, d.ci_hi - d.log2_ratio],
                    fmt="o", color=cols[g], ecolor="#888", capsize=2)
        ax.axvline(0, color="k", lw=.6)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d.lemma)
        ax.set_title(f"{g}: top-15 by G²")
        ax.set_xlabel("log2 (learner / L1 Korean rate)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_rq1_distribution.png", dpi=150)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    tok, args = load()
    tok = add_error_flags(tok)
    tok.to_pickle(WORK / "tokens_flagged.pkl")
    proxy_validation(tok)
    wt = writer_table(tok, args)
    pd.to_pickle(wt, WORK / "writer_table.pkl")
    meas, h1, h3, idx = run_measures(wt, rng)
    meas.to_csv(RESULTS / "rq1_measures.csv", index=False)
    h1.to_csv(RESULTS / "rq1_h1_pairwise_abs_lor.csv", index=False)
    h3.to_csv(RESULTS / "rq1_h3_interface_vs_narrow.csv", index=False)
    dist, clsdf, sp, corr = distribution(tok, idx)
    dist.to_csv(RESULTS / "rq1_distribution_by_lemma.csv", index=False)
    clsdf.to_csv(RESULTS / "rq1_distribution_by_class.csv", index=False)
    sp.to_csv(RESULTS / "rq1_distribution_group_spearman.csv", index=False)
    corr.to_csv(RESULTS / "rq1_distribution_freq_deltaP.csv", index=False)
    col = collocations(tok, rng)
    col.to_csv(RESULTS / "rq1_collocations_stacking.csv", index=False)
    top_collocations(tok).to_csv(RESULTS / "rq1_top_collocations.csv", index=False)
    figures(meas, dist)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(meas[["measure", "group", "num", "den", "estimate", "ci_lo", "ci_hi", "lor"]].round(3).to_string())
    print(h3.round(3).to_string())
    print(sp.round(3), corr.round(3))
    print(col.round(3).to_string())


if __name__ == "__main__":
    main()
