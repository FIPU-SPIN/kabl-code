#!/usr/bin/env python3
"""Validate coder / LLM CSV files BEFORE analysis.

Usage:
  python scripts/validate_csv.py                 # all files in data/human and data/llm_raw
  python scripts/validate_csv.py path/to/a.csv   # specific files

Checks: required columns, 127 rows, keys (stage+code) identical to the original input,
no duplicates, Bloom in 1..6, ABC exactly one of the six words, row order.
ERRORS must be fixed (re-run the model or fix the file); WARNINGS are informational.
Exit code 1 if any file has errors.
"""
import sys
from pathlib import Path

from common import (ABC_TYPES, INPUT_DIR, N_EXPECTED, clean_abc, clean_bloom,
                    list_rater_files, make_key, read_coding_csv, read_original)


def shuffled_keys():
    for p in sorted(INPUT_DIR.glob("input_shuffled_seed*.tsv")):
        rows = [l.split("\t") for l in p.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
        return [make_key(r[0], r[2]) for r in rows]
    return None


def validate(path, orig_keys, shuf_keys):
    errors, warns = [], []
    try:
        df = read_coding_csv(path)
    except Exception as e:  # noqa: BLE001
        return [f"cannot read file: {e}"], []
    warns.extend("repaired in memory: " + r for r in df.attrs.get("repairs", []))
    need = {"stage", "grade", "code", "bloom", "abc"}
    missing = need - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)} (found: {list(df.columns)})"], []

    if len(df) != N_EXPECTED:
        errors.append(f"{len(df)} rows, expected {N_EXPECTED}")
    dup = df["key"][df["key"].duplicated()].tolist()
    if dup:
        errors.append(f"duplicate keys: {dup[:5]}{' ...' if len(dup) > 5 else ''}")
    miss = sorted(set(orig_keys) - set(df["key"]))
    extra = sorted(set(df["key"]) - set(orig_keys))
    if miss:
        errors.append(f"{len(miss)} outcomes missing, e.g. {miss[:5]}")
    if extra:
        errors.append(f"{len(extra)} unknown keys, e.g. {extra[:5]}")

    keys = df["key"].tolist()
    if keys != orig_keys and keys != shuf_keys and not miss and not extra and not dup:
        warns.append("row order differs from original and from shuffled input "
                     "(OK: analysis aligns by stage+code)")

    bad_b, fix_b, bad_a, fix_a = [], [], [], []
    for k, b, a in zip(df["key"], df["bloom"], df["abc"]):
        _, st = clean_bloom(b)
        if st in ("invalid", "empty"):
            bad_b.append((k, b))
        elif st == "normalized":
            fix_b.append((k, b))
        _, st = clean_abc(a)
        if st in ("invalid", "empty"):
            bad_a.append((k, a))
        elif st == "normalized":
            fix_a.append((k, a))
    if bad_b:
        errors.append(f"{len(bad_b)} invalid/empty Bloom values, e.g. {bad_b[:5]}")
    if bad_a:
        errors.append(f"{len(bad_a)} invalid/empty ABC values (allowed: {ABC_TYPES}), e.g. {bad_a[:5]}")
    if fix_b:
        warns.append(f"{len(fix_b)} Bloom values need normalisation (e.g. '4.0'): {fix_b[:3]}")
    if fix_a:
        warns.append(f"{len(fix_a)} ABC values differ only in case/spacing/punctuation "
                     f"(merge_codings.py normalises and logs them): {fix_a[:3]}")
    return errors, warns


def main():
    orig_keys = read_original()["key"].tolist()
    shuf_keys = shuffled_keys()
    if len(sys.argv) > 1:
        files = [(Path(p), None) for p in sys.argv[1:]]
    else:
        files = list_rater_files()
    if not files:
        print("No CSV files found in data/human or data/llm_raw.")
        return 1
    n_err = 0
    for path, _ in files:
        errors, warns = validate(path, orig_keys, shuf_keys)
        print(f"[{'FAIL' if errors else 'OK  '}] {path.name}")
        for e in errors:
            print(f"        ERROR  : {e}")
        for w in warns:
            print(f"        warning: {w}")
        n_err += bool(errors)
    print(f"\n{len(files) - n_err}/{len(files)} files passed.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())