# L2 Korean postposition (josa) analysis

Code for "An Analysis of Postposition Use in Second-Language Korean". Data: `data/` (UD KSL and KAIST).

    pip install -r requirements.txt
    cd src && python run_all.py     # extract -> RQ1 -> RQ2 -> RQ3 -> validation sheets (~10 min)

- `src/extract.py` postposition extraction (reproduces Table 2); `rq1.py`, `rq2.py` (+`glmm.py`), `rq3.py`; `validation.py` manual-validation workflow.
- Outputs are in `results/` (CSVs and figures). `DECISIONS.md` lists every implementation choice and the open questions.
