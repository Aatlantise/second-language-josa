"""Postposition information extraction from the KSL and KAIST UD treebanks.

Produces three pickled tables in work/:
  tokens.pkl  one row per token (all corpora), with postposition annotation
  args.pkl    one row per argument root (subjects / objects), with features
  meta.pkl    extraction diagnostics (counts of skipped/misaligned items, ...)

Every rule here is documented in DECISIONS.md.
"""
import glob
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

from common import (ARG_CODE, CLASS_OF, DATA, WORK, canon_lemma, coda_class,
                    expected_allomorph_ok, pp_tag)

NOMINAL_UPOS = {"NOUN", "PRON", "PROPN", "NUM"}
SUBJ_RELS = {"nsubj", "nsubj:pass", "nsubj:outer", "csubj"}
BOUND_NOUNS = {"수", "것"}              # excluded argument roots (paper, RQ1 measures)
NNB_TAGS = {"NNB", "nbn"}


# ------------------------------------------------------------------ reading
def read_conllu(paths):
    """Yield (meta, [rows]) for each sentence; multiword-token/empty-node lines skipped."""
    for p in paths:
        meta, rows = {}, []
        with open(p, encoding="utf8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("#"):
                    m = re.match(r"# (\w+) = (.*)", line)
                    if m:
                        meta[m.group(1)] = m.group(2)
                elif not line:
                    if rows:
                        yield meta, rows
                    meta, rows = {}, []
                else:
                    c = line.split("\t")
                    if len(c) != 10 or "-" in c[0] or "." in c[0]:
                        continue
                    rows.append(c)
        if rows:
            yield meta, rows


def sentence_info(corpus, sid):
    """group, writer, essay for a sentence id; group None => not used (no L1 label)."""
    if corpus == "KSL":
        if not sid.startswith("ARG-"):
            return None, None, None
        parts = sid.split("_")
        return ARG_CODE[parts[0][4:]], parts[0] + "_" + parts[1], parts[2]
    return "L1 Korean", sid.rsplit("-s", 1)[0], None


# --------------------------------------------------------- token annotation
def annotate_token(corpus, form, lemma, xpos, diag):
    """Postposition fields for one token."""
    tags = xpos.split("+")
    morphs = lemma.split("+")
    if len(tags) != len(morphs):                     # lemma contains '+' (symbols) etc.
        diag["misaligned_tokens"] += 1
        morphs = None
    pp_idx = [i for i, t in enumerate(tags) if pp_tag(corpus, t)]
    out = dict(n_pp=len(pp_idx), pp_tags="", pp_classes="", pp_lemmas="", pp_raw="",
               stem="", stem_coda=None, bound_noun=False, detached=False,
               allomorph_eval=0, allomorph_err=0, first_pp="", first_pp_class="")
    if not pp_idx:
        # root frequency / coda still needed for omitted arguments
        if morphs:
            out["stem"] = "+".join(morphs)
            out["stem_coda"] = coda_class(morphs[-1][-1:] if morphs[-1] else "")
            out["bound_noun"] = tags[-1] in NNB_TAGS and morphs[-1] in BOUND_NOUNS
        return out
    sej = [pp_tag(corpus, tags[i]) for i in pp_idx]
    out["pp_tags"] = "+".join(sej)
    out["pp_classes"] = "+".join(CLASS_OF[t] for t in sej)
    out["first_pp_class"] = CLASS_OF[sej[0]]
    out["detached"] = (pp_idx[0] == 0 and len(pp_idx) == len(tags))
    if morphs is None:
        return out
    raw = [morphs[i] for i in pp_idx]
    out["pp_raw"] = "+".join(raw)
    out["pp_lemmas"] = "+".join(canon_lemma(m) for m in raw)
    out["first_pp"] = canon_lemma(raw[0])
    first = pp_idx[0]
    if first > 0:
        stem_m = morphs[:first]
        out["stem"] = "+".join(stem_m)
        out["bound_noun"] = tags[first - 1] in NNB_TAGS and morphs[first - 1] in BOUND_NOUNS
        prev = stem_m[-1][-1:] if stem_m[-1] else ""
    else:
        prev = ""
    out["stem_coda"] = coda_class(prev)
    # allomorph check: each postposition vs the final sound of the morpheme before it
    ev = err = 0
    for k, i in enumerate(pp_idx):
        if i == 0:
            continue                                  # detached postposition: no stem to check
        prev = morphs[i - 1][-1:] if morphs[i - 1] else ""
        tail = "".join(morphs[i:])                    # postpositions are always word-final
        if not (form.endswith(tail) and len(form) > len(tail) and form[-len(tail) - 1] == prev):
            diag["allomorph_skipped_contracted_or_typo"] += 1
            continue                                  # contraction/typo: surface != lemma string
        ok = expected_allomorph_ok(raw[k], coda_class(prev))
        if ok is not None:
            ev += 1
            err += (not ok)
    out["allomorph_eval"], out["allomorph_err"] = ev, err
    return out


def build_tokens():
    diag = Counter()
    recs = []
    sent_no = 0
    for corpus, pat in (("KSL", "ko_ksl-ud-*.conllu"), ("KAIST", "ko_kaist-ud-*.conllu")):
        for path in sorted(glob.glob(str(DATA / pat))):
            split = re.search(r"-(train|dev|test)\.", path).group(1)
            for meta, rows in read_conllu([path]):
                sid = meta["sent_id"]
                group, writer, essay = sentence_info(corpus, sid)
                if group is None:
                    diag["ksl_sentences_without_L1_label"] += 1
                    continue
                sent_no += 1
                n = len(rows)
                for c in rows:
                    ann = annotate_token(corpus, c[1], c[2], c[4], diag)
                    comment = re.search(r"Comment=([^|]*)", c[9])
                    recs.append(dict(
                        corpus=corpus, split=split, group=group, writer=writer, essay=essay,
                        sent_id=sid, sno=sent_no, i=int(c[0]), form=c[1], lemma=c[2], upos=c[3],
                        xpos=c[4], head=int(c[6]), deprel=c[7],
                        typo="Typo=Yes" in c[5],
                        comment=comment.group(1) if comment else "",
                        sent_len=n, **ann))
    df = pd.DataFrame.from_records(recs)
    return df, diag


# ------------------------------------------------------ argument-root table
def build_args(tok, diag):
    """One row per argument root. Rules (see DECISIONS.md):
       * nominal UPOS (NOUN/PRON/PROPN/NUM), excluding bound nouns 수/것
       * subject: nsubj*/csubj, or `dislocated` whose head has no other subject and whose
         postposition stack holds no non-nominative case postposition (KAIST labels most
         topic-marked subjects `dislocated`)
       * object: obj
       * obliques/conjuncts are excluded
    """
    rows = []
    for sno, s in tok.groupby("sno", sort=False):
        s = s.set_index("i", drop=False)
        idx = s.index.tolist()
        deprel = s["deprel"].to_dict()
        head = s["head"].to_dict()
        has_subj = defaultdict(bool)
        for i in idx:
            if deprel[i] in SUBJ_RELS:
                has_subj[head[i]] = True

        def clause_type(h):
            seen = 0
            while h in deprel and deprel[h] == "conj" and seen < 20:
                h, seen = head[h], seen + 1
            return "main" if (h == 0 or deprel.get(h) in ("root", "parataxis")) else "sub"

        # detached postposition tokens attached to a root (e.g. "그 것 은")
        detached = {}
        for i in idx:
            r = s.loc[i]
            if r.detached and (i - 1) in deprel and head[i] == i - 1:
                detached[i - 1] = r
        cand = []
        for i in idx:
            r = s.loc[i]
            if r.upos not in NOMINAL_UPOS or r.bound_noun:
                continue
            d = r.deprel
            if d in SUBJ_RELS:
                role, disl = "subj", False
            elif d == "obj":
                role, disl = "obj", False
            elif d == "dislocated" and not has_subj[head[i]]:
                role, disl = "subj", True
            else:
                continue
            cand.append((i, role, disl, r))
        subj_pos = defaultdict(list)
        obj_pos = defaultdict(list)
        for i, role, disl, r in cand:
            (subj_pos if role == "subj" else obj_pos)[r["head"]].append(i)
        for i, role, disl, r in cand:
            h = r["head"]
            tags, lems, n_pp, first = r.pp_tags, r.pp_lemmas, r.n_pp, r.first_pp
            det = False
            if n_pp == 0 and i in detached:
                dr = detached[i]
                tags, lems, n_pp, first = dr.pp_tags, dr.pp_lemmas, dr.n_pp, dr.first_pp
                det = True
                diag["args_marked_by_detached_postposition"] += 1
            if disl:
                # a dislocated root carrying a non-nominative case postposition is an oblique
                if any(t in ("JKB", "JKG", "JKO", "JKC", "JKQ", "JKV") for t in tags.split("+") if t):
                    diag["dislocated_excluded_oblique"] += 1
                    continue
            order = "single"
            if subj_pos[h] and obj_pos[h]:
                ms, mo = min(subj_pos[h]), min(obj_pos[h])
                order = "SOV" if ms < mo < h else "OSV" if mo < ms else "other"
            lemma_set = lems.split("+") if lems else []
            if role == "subj":
                cat = ("TOP" if "은/는" in lemma_set else "NOM" if "JKS" in tags.split("+")
                       else "OTHER" if n_pp else "NONE")
            else:
                cat = "ACC" if "JKO" in tags.split("+") else "OTHER" if n_pp else "NONE"
            rows.append(dict(
                corpus=r.corpus, group=r.group, writer=r.writer, essay=r.essay, sno=sno,
                sent_id=r.sent_id, i=i, form=r.form, role=role, dislocated=disl,
                upos=r.upos, deprel=r.deprel, head=h,
                head_dist=abs(h - i) if h else 0, order=order, clause=clause_type(h),
                sent_len=r.sent_len, stem=r.stem, coda=r.stem_coda, marked=int(n_pp > 0),
                category=cat, n_pp=n_pp, pp_tags=tags, pp_lemmas=lems, detached=det,
                split=r.split))
    return pd.DataFrame(rows)


def main():
    tok, diag = build_tokens()
    # --- reproduce the paper's Table 2 (sanity check)
    t2 = (tok.assign(has_pp=tok.n_pp > 0)
             .groupby(["corpus", "group"])
             .agg(learners_or_docs=("writer", "nunique"), sentences=("sno", "nunique"),
                  tokens=("i", "size"), pp_tokens=("has_pp", "sum")))
    print(t2)
    args = build_args(tok, diag)
    tok.to_pickle(WORK / "tokens.pkl")
    args.to_pickle(WORK / "args.pkl")
    pd.to_pickle(dict(diag), WORK / "meta.pkl")
    t2.to_csv(WORK.parent / "results" / "table2_reproduction.csv")
    print(dict(diag))
    print(args.groupby(["group", "role"]).agg(n=("i", "size"), omitted=("marked", lambda x: 1 - x.mean())))


if __name__ == "__main__":
    main()
