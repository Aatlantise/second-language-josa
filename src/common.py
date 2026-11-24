"""Shared constants and helpers for the L2-Korean postposition (josa) analysis."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORK = ROOT / "work"
RESULTS = ROOT / "results"
for _d in (WORK, RESULTS):
    _d.mkdir(exist_ok=True)

SEED = 20261009
N_BOOT = 1000          # paper: 1,000 bootstrap resamples
L2_GROUPS = ["Chinese", "Czech", "English", "Japanese"]
BASELINE = "L1 Korean"
ALL_GROUPS = L2_GROUPS + [BASELINE]
ARG_CODE = {"CHN": "Chinese", "CZH": "Czech", "ENG": "English", "JPN": "Japanese"}

# ---------------------------------------------------------------- tagsets
# KSL uses Sejong tags (JKS, JKO, JX, ...); KAIST uses its own lower-case tags.
# Everything is mapped to the Sejong-style name so both corpora share one scheme.
KAIST_TO_SEJONG = {
    "jcs": "JKS", "jcc": "JKC", "jcm": "JKG", "jco": "JKO", "jca": "JKB",
    "jcv": "JKV", "jcr": "JKQ", "jct": "JKB",   # jct = comitative -> adverbial case
    "jcj": "JC", "jxt": "JX", "jxc": "JX", "jxf": "JX",
}
CASE_TAGS = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ"}
CLASS_OF = {**{t: "case" for t in CASE_TAGS}, "JX": "auxiliary", "JC": "conjunctive"}


def pp_tag(corpus, xpos_tag):
    """Return the Sejong-style postposition tag for one morpheme tag, or None."""
    if corpus == "KAIST":
        return KAIST_TO_SEJONG.get(xpos_tag)          # 'jp' (copula) -> None
    if xpos_tag.startswith("JK"):
        return xpos_tag if xpos_tag in CASE_TAGS else "JKB" if xpos_tag == "JK" else None
    if xpos_tag.startswith("JX"):
        return "JX"
    if xpos_tag.startswith("JC"):
        return "JC"
    return None


# ------------------------------------------------- allomorph tables
# (form after a consonant-final stem, form after a vowel-final stem)
ALLOMORPH_PAIRS = [
    ("이", "가"), ("을", "를"), ("은", "는"), ("과", "와"), ("이랑", "랑"),
    ("이나", "나"), ("이든", "든"), ("이든지", "든지"), ("이라도", "라도"),
    ("이라고", "라고"), ("이야", "야"), ("이나마", "나마"), ("이라야", "라야"),
    ("이면", "면"), ("이며", "며"), ("이고", "고"),
]
# 으로-type: 으로 after non-ㄹ consonant, 로 after vowel or ㄹ
EURO_FAMILY = {"으로": "로", "으로서": "로서", "으로써": "로써", "으로부터": "로부터"}

CANON = {}
for _c, _v in ALLOMORPH_PAIRS:
    CANON[_c] = CANON[_v] = f"{_c}/{_v}"
for _c, _v in EURO_FAMILY.items():
    CANON[_c] = CANON[_v] = f"{_c}/{_v}"
# JKS/JKC "이/가" only; a bare 이/고 etc. are ambiguous with copula morphs but the
# copula is never tagged as a postposition, so the table is safe.


def canon_lemma(lemma):
    return CANON.get(lemma, lemma)


def coda_class(ch):
    """'V' vowel-final, 'L' ㄹ-final, 'C' other consonant, None if not Hangul/jamo."""
    if not ch:
        return None
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:
        c = (o - 0xAC00) % 28
        return "V" if c == 0 else "L" if c == 8 else "C"
    if 0x3131 <= o <= 0x314E:          # bare consonant jamo (e.g. nominaliser ㅁ)
        return "L" if ch == "ㄹ" else "C"
    if 0x314F <= o <= 0x3163:
        return "V"
    return None


def expected_allomorph_ok(form, prev_coda):
    """Is allomorph `form` correct after a stem with final-sound class prev_coda?
    Returns None when the form is not allomorphic or the stem is not evaluable."""
    if prev_coda is None:
        return None
    for c, v in ALLOMORPH_PAIRS:
        if form == c:
            return prev_coda != "V"
        if form == v:
            return prev_coda == "V"
    for c, v in EURO_FAMILY.items():
        if form == c:
            return prev_coda == "C"
        if form == v:
            return prev_coda in ("V", "L")
    return None
