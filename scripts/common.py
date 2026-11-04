"""Shared helpers for the KABL-ID coding pipeline."""
import csv
import io
import os
import re
from pathlib import Path

import pandas as pd

# Repository root. Can be overridden (e.g. for a dry run on synthetic data):
#   KABL_ROOT=/tmp/demo python scripts/analysis.py
ROOT = Path(os.environ.get("KABL_ROOT", Path(__file__).resolve().parent.parent))

N_EXPECTED = 127
ABC_TYPES = ["Acquisition", "Discussion", "Investigation",
             "Collaboration", "Practice", "Production"]
BLOOM_LEVELS = [1, 2, 3, 4, 5, 6]

KEY_COLS = ["stage", "grade", "code"]

# Column names are normalised (lower-case, stripped) and then mapped through this.
COLUMN_ALIASES = {
    "bloom_razina_1_do_6": "bloom",
    "bloom": "bloom",
    "abc_tip": "abc",
    "abc": "abc",
    "napomena": "napomena",
    "napomena_ako_nisi_sigurna": "napomena",
}

INPUT_DIR = ROOT / "data" / "input"
HUMAN_DIR = ROOT / "data" / "human"
LLM_DIR = ROOT / "data" / "llm_raw"
PROC_DIR = ROOT / "data" / "processed"
PROTOCOL_DIR = ROOT / "protocol"
TABLES_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"


def make_key(stage, code):
    return f"{str(stage).strip().upper()}|{str(code).strip()}"


def read_original():
    """Original input (defines canonical row set and order)."""
    df = pd.read_csv(INPUT_DIR / "input_original.tsv", sep="\t", dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
    df["key"] = [make_key(s, c) for s, c in zip(df["stage"], df["code"])]
    if df["key"].duplicated().any():
        raise ValueError("Duplicate stage+code keys in input_original.tsv")
    return df


_WRAPPED = re.compile(r'^"(.*)"[;,\t ]*$')


def _unwrap_quoted_lines(lines):
    """Undo 'whole line in one quoted cell'.

    Happens when CSV text is pasted into a single Excel column and saved as CSV:
    every line becomes  "a,b,c"  (inner quotes doubled). Detected on the header line.
    """
    m = _WRAPPED.match(lines[0].strip())
    if not m or sum(m.group(1).count(d) for d in (",", ";", "\t")) < 3:
        return lines
    out = []
    for ln in lines:
        m = _WRAPPED.match(ln.strip())
        out.append(m.group(1).replace('""', '"') if m else ln)
    return out


_CODE_PREFIX = re.compile(r"^\s*(OS|SS)\s*(\d+)\s*([A-Da-d]\.\d.*)$", re.IGNORECASE)


def read_coding_csv(path):
    """Read a coder/LLM CSV leniently. Never modifies the file on disk.

    Repairs applied in memory (listed in df.attrs["repairs"], shown as warnings by
    validate_csv.py and written to data/processed/normalization_log.csv):
      * BOM, markdown code fences (```), blank lines
      * comma / semicolon / tab delimiters (Excel in Croatian locale saves with ';')
      * every line wrapped in quotes (CSV pasted into one Excel column)
      * unquoted commas inside the LAST column (napomena) -> extra fields merged back
      * header line repeated in the middle (e.g. pasted continuation) -> dropped
      * incomplete rows with empty code (model output cut off mid-row) -> dropped
      * code column containing stage+grade prefix ('OS1A.1.1') -> split into stage/grade/code
    Returns DataFrame with columns stage, grade, code, bloom, abc, napomena, key (strings).
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("```")]
    if not lines:
        raise ValueError(f"{path}: empty file")
    lines = _unwrap_quoted_lines(lines)
    delim = max([",", ";", "\t"], key=lines[0].count)
    repairs = []

    rows = list(csv.reader(lines, delimiter=delim, skipinitialspace=True))
    header = [COLUMN_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in rows[0]]
    n = len(header)
    last_is_note = header[-1] == "napomena"
    fixed, n_merge, n_hdr = [], 0, 0
    for r in rows[1:]:
        if [c.strip().lower() for c in r[:3]] == ["stage", "grade", "code"]:
            n_hdr += 1
            continue
        if len(r) > n:
            if not last_is_note:
                raise ValueError(f"row has {len(r)} fields, expected {n}: {r[:4]}")
            r = r[:n - 1] + [delim.join(r[n - 1:])]
            n_merge += 1
        elif len(r) < n:
            r = r + [""] * (n - len(r))
        fixed.append(r)
    df = pd.DataFrame(fixed, columns=header, dtype=str)
    if "napomena" not in df.columns:
        df["napomena"] = ""
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    if n_merge:
        repairs.append(f"{n_merge} rows had unquoted commas in napomena (merged back into one field)")
    if n_hdr:
        repairs.append(f"{n_hdr} repeated header row(s) dropped")

    if {"stage", "grade", "code"} <= set(df.columns):
        empty = df["code"] == ""
        if empty.any():
            repairs.append(f"{int(empty.sum())} incomplete row(s) with empty code dropped "
                           f"(output probably cut off): {df.loc[empty, 'stage'].tolist()}")
            df = df[~empty].reset_index(drop=True)
        ex = df["code"].str.extract(_CODE_PREFIX)
        pref = ex[0].notna()
        if pref.any():
            df.loc[pref, "stage"] = ex.loc[pref, 0].str.upper()
            df.loc[pref, "grade"] = ex.loc[pref, 1]
            df.loc[pref, "code"] = ex.loc[pref, 2]
            repairs.append(f"{int(pref.sum())} rows had stage+grade prefix in code column "
                           f"(e.g. 'OS1A.1.1' -> OS / 1 / A.1.1); split into separate columns")
    if {"stage", "code"} <= set(df.columns):
        df["key"] = [make_key(s_, c_) for s_, c_ in zip(df["stage"], df["code"])]
    df.attrs["repairs"] = repairs
    return df


def clean_bloom(raw):
    """Return (int or None, status) with status in ok / normalized / invalid / empty."""
    s = str(raw).strip()
    if s == "":
        return None, "empty"
    try:
        f = float(s.replace(",", "."))
    except ValueError:
        return None, "invalid"
    if f.is_integer() and 1 <= int(f) <= 6:
        return int(f), ("ok" if s == str(int(f)) else "normalized")
    return None, "invalid"


def clean_abc(raw):
    """Return (canonical label or None, status). Only case/whitespace/punctuation are fixed."""
    s = str(raw).strip()
    if s == "":
        return None, "empty"
    if s in ABC_TYPES:
        return s, "ok"
    s2 = s.strip(" .,;:\"'*`").lower()
    lookup = {t.lower(): t for t in ABC_TYPES}
    if s2 in lookup:
        return lookup[s2], "normalized"
    return None, "invalid"


def list_rater_files():
    """[(path, rater_type)] for all human and LLM raw files."""
    out = [(p, "human") for p in sorted(HUMAN_DIR.glob("*.csv"))]
    out += [(p, "llm") for p in sorted(LLM_DIR.glob("*.csv"))]
    return out


def rater_meta(path, rater_type):
    stem = Path(path).stem
    if rater_type == "human":
        return {"rater_id": stem, "rater_type": "human", "model": "human", "variant": stem}
    parts = stem.split("_", 1)
    return {"rater_id": stem, "rater_type": "llm", "model": parts[0].lower(),
            "variant": parts[1].lower() if len(parts) > 1 else "run1"}