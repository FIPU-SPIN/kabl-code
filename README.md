# KABL-ID: data and analysis code

Coding data and scripts for the inter-rater agreement analysis (human coders vs. LLMs)
of the Croatian Informatics curriculum outcomes (NN 22/2018). Manuscript: see the
`kabl-id-latex` repository.

## Pipeline

```bash
pip install -r requirements.txt

python scripts/make_inputs.py          # shuffled input + ready-to-paste prompts (seed 42)
# ... collect codings -> data/human/coder1.csv, coder2.csv ; data/llm_raw/<model>_run1..3.csv, <model>_shuffled.csv
python scripts/validate_csv.py         # check every file (127 rows, valid Bloom/ABC values, keys)
python scripts/merge_codings.py        # -> data/processed/  (raw files are never modified)
python scripts/analysis.py --n-boot 2000 --seed 42   # -> results/tables, results/figures
```

Dry run on synthetic data (does not touch real data):

```bash
python tests/make_synthetic_data.py /tmp/kabl_demo
KABL_ROOT=/tmp/kabl_demo python scripts/validate_csv.py
KABL_ROOT=/tmp/kabl_demo python scripts/merge_codings.py
KABL_ROOT=/tmp/kabl_demo python scripts/analysis.py --n-boot 200
```

## File naming
- Humans: `coder1.csv`, `coder2.csv` (anonymous)
- LLMs: `<model>_run1.csv`, `<model>_run2.csv`, `<model>_run3.csv`, `<model>_shuffled.csv`
  (e.g. `claude_run1.csv`, `chatgpt_shuffled.csv`); the part before the first `_` is the model name.

## Statistics
Cohen's kappa, quadratic-weighted kappa (Bloom only), Gwet's AC1, percent agreement,
Krippendorff's alpha (nominal / ordinal / interval); 95% percentile-bootstrap CIs
(items resampled, fixed seed). Model consensus = majority of the model's three base runs
(ties are left missing and counted in `results/tables/consensus_info.csv`).
The order-shuffled run is reported separately and is not part of the consensus.

## Licences
Code: MIT. Data: CC BY 4.0.
