"""RQ3: can a learner's L1 be predicted from postposition use alone, and which measures
carry the information?   Multinomial L2-regularised logistic regression, stratified 5-fold CV
repeated 10 times; compared with the majority-class baseline and a permutation test.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import L2_GROUPS, RESULTS, SEED, WORK

warnings.filterwarnings("ignore")
C_REG = 0.1                      # fixed a priori (strong shrinkage: ~38 features, 157 learners)
N_REPEATS = 10
N_PERM = 1000
TOP_LEMMAS = 30
# typological profile: (overt case marking, topic marker, optional particle omission)
PROFILE = {"Chinese": (0, 0, 0), "English": (0, 0, 0), "Czech": (1, 0, 0), "Japanese": (1, 1, 1)}
# competition-vs-cooperation prompt, assuming essay number = prompt order given in the paper
SHARED_ESSAY = {"Chinese": "2", "Czech": "3", "English": "3", "Japanese": "3"}


def top_lemmas(tok):
    t = tok[(tok.group.isin(L2_GROUPS)) & (tok.pp_lemmas != "")]
    return t.pp_lemmas.str.split("+").explode().value_counts().head(TOP_LEMMAS).index.tolist()


def learner_features(tok, args, lemmas):
    tok = tok[tok.group.isin(L2_GROUPS)]
    args = args[args.group.isin(L2_GROUPS)]
    writers = tok.drop_duplicates("writer").set_index("writer")["group"].sort_index()
    ntok = tok.groupby("writer").size().reindex(writers.index)
    F = pd.DataFrame(index=writers.index)
    ex = tok[tok.pp_lemmas != ""].assign(l=lambda d: d.pp_lemmas.str.split("+")).explode("l")
    cnt = ex.groupby(["writer", "l"]).size().unstack(fill_value=0).reindex(index=writers.index,
                                                                           columns=lemmas, fill_value=0)
    for l in lemmas:
        F["dist:" + l] = 1000 * cnt[l] / ntok
    for role in ("subj", "obj"):
        a = args[args.role == role].groupby("writer").marked.agg(["size", "mean"])
        F[f"omission:{role}"] = (1 - a["mean"]).reindex(F.index)
    a = args[(args.role == "subj") & args.category.isin(["TOP", "NOM"])]
    F["choice:topic_among_marked_subj"] = (a.category == "TOP").groupby(a.writer).mean().reindex(F.index)
    t = tok[tok.n_pp > 0]
    F["stacking:rate"] = (t.n_pp >= 2).groupby(t.writer).mean().reindex(F.index)
    e = tok.groupby("writer")[["allomorph_err", "allomorph_eval"]].sum()
    F["errors:allomorph"] = (e.allomorph_err / e.allomorph_eval.replace(0, np.nan)).reindex(F.index)
    s = tok[tok.subst_eval]
    F["errors:substitution_proxy"] = s.subst_flag.groupby(s.writer).mean().reindex(F.index)
    i = tok[tok.invalid_eval]
    F["errors:invalid_form"] = i.invalid_flag.groupby(i.writer).mean().reindex(F.index)
    return F, writers


def make_model(c=C_REG):
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                         LogisticRegression(C=c, max_iter=3000))


def cv_eval(X, y, model_fn, n_repeats=N_REPEATS, seed=SEED, return_pred=False):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    accs, f1s, preds = [], [], []
    fold_pred = np.empty((n_repeats, len(y)), dtype=object)
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        m = model_fn().fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
        fold_pred[k // 5, te] = p
    rep_acc = [np.mean(accs[r * 5:(r + 1) * 5]) for r in range(n_repeats)]
    rep_f1 = [np.mean(f1s[r * 5:(r + 1) * 5]) for r in range(n_repeats)]
    out = dict(acc=np.mean(rep_acc), acc_sd=np.std(rep_acc), f1=np.mean(rep_f1), f1_sd=np.std(rep_f1))
    return (out, fold_pred) if return_pred else out


def perm_test(X, y, observed_acc, rng, n_perm=N_PERM, repeats=2):
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(y)
        null[b] = cv_eval(X, yp, make_model, n_repeats=repeats, seed=int(rng.integers(1e9)))["acc"]
    return null, (1 + (null >= observed_acc).sum()) / (n_perm + 1)


def confusion(y, fold_pred):
    labs = L2_GROUPS
    C = np.zeros((4, 4))
    for r in range(fold_pred.shape[0]):
        for t, p in zip(y, fold_pred[r]):
            C[labs.index(t), labs.index(p)] += 1
    return pd.DataFrame(C / C.sum(1, keepdims=True), index=labs, columns=labs)


def typology_check(cm):
    rows = []
    labs = L2_GROUPS
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = labs[i], labs[j]
            dist = sum(x != y for x, y in zip(PROFILE[a], PROFILE[b]))
            rows.append(dict(pair=f"{a}-{b}", typological_distance=dist,
                             confusion_rate=(cm.loc[a, b] + cm.loc[b, a]) / 2))
    df = pd.DataFrame(rows)
    rho = spearmanr(df.typological_distance, df.confusion_rate)[0]
    return df, rho


def groups_of(F):
    out = {}
    for c in F.columns:
        out.setdefault(c.split(":")[0], []).append(c)
    return out


def run(F, writers, tag, rng, do_perm=True):
    X, y = F.values, writers.values
    res = {}
    obs, fold_pred = cv_eval(X, y, make_model, return_pred=True)
    maj = cv_eval(X, y, lambda: DummyClassifier(strategy="most_frequent"))
    obs.update(model="logistic", n=len(y))
    res["main"] = [obs, dict(model="majority_class", n=len(y), **maj)]
    if do_perm:
        null, p = perm_test(X, y, obs["acc"], rng)
        res["main"][0].update(perm_p=p, perm_null_mean=null.mean(), perm_null_p95=np.percentile(null, 95))
    cm = confusion(y, fold_pred)
    tc, rho = typology_check(cm)
    # sensitivity to the regularisation strength (reported, not used for selection)
    sens = []
    for c in (0.01, 0.1, 1, 10):
        r = cv_eval(X, y, lambda c=c: make_model(c), n_repeats=3)
        sens.append(dict(C=c, **r))
    # feature-group ablation (drop one group / keep only that group)
    ab = []
    for g, cols in groups_of(F).items():
        keep = [i for i, c in enumerate(F.columns) if c not in cols]
        only = [i for i, c in enumerate(F.columns) if c in cols]
        r1 = cv_eval(X[:, keep], y, make_model, n_repeats=N_REPEATS)
        r2 = cv_eval(X[:, only], y, make_model, n_repeats=N_REPEATS)
        ab.append(dict(group=g, n_features=len(cols), acc_without=r1["acc"], f1_without=r1["f1"],
                       acc_drop=obs["acc"] - r1["acc"], f1_drop=obs["f1"] - r1["f1"],
                       acc_only=r2["acc"], f1_only=r2["f1"]))
    pd.DataFrame(res["main"]).to_csv(RESULTS / f"rq3_{tag}_accuracy.csv", index=False)
    cm.to_csv(RESULTS / f"rq3_{tag}_confusion.csv")
    tc.to_csv(RESULTS / f"rq3_{tag}_typology_check.csv", index=False)
    pd.DataFrame(sens).to_csv(RESULTS / f"rq3_{tag}_regularisation_sensitivity.csv", index=False)
    pd.DataFrame(ab).to_csv(RESULTS / f"rq3_{tag}_ablation.csv", index=False)
    print(tag, pd.DataFrame(res["main"]).round(3).to_string())
    print(cm.round(2))
    print(tc.round(3), "spearman(distance, confusion) =", round(rho, 3))
    print(pd.DataFrame(sens).round(3))
    print(pd.DataFrame(ab).round(3))
    return cm


def main():
    rng = np.random.default_rng(SEED)
    tok = pd.read_pickle(WORK / "tokens_flagged.pkl")
    args = pd.read_pickle(WORK / "args.pkl")
    lemmas = top_lemmas(tok)
    F, w = learner_features(tok, args, lemmas)
    F.assign(L1=w).to_csv(RESULTS / "rq3_learner_features.csv")
    run(F, w, "all_essays", rng)
    # shared prompt rerun (competition vs cooperation) -- essay-number mapping is an assumption
    keep_t = tok[tok.apply(lambda r: r.group in SHARED_ESSAY and r.essay == SHARED_ESSAY[r.group], axis=1)]
    keep_a = args[args.apply(lambda r: r.group in SHARED_ESSAY and r.essay == SHARED_ESSAY[r.group], axis=1)]
    F2, w2 = learner_features(keep_t, keep_a, lemmas)
    print("shared-prompt learners per group:", w2.value_counts().to_dict())
    run(F2, w2, "shared_prompt", rng)
    return None


if __name__ == "__main__":
    main()
