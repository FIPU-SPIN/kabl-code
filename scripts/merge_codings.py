#!/usr/bin/env python3
"""Merge all raw codings into analysis-ready tables (raw files are never modified).

Reads : data/human/*.csv, data/llm_raw/*.csv, data/input/input_original.tsv
Writes: data/processed/all_codings_long.csv
        data/processed/bloom_wide.csv, abc_wide.csv   (rows = outcomes, columns = raters)
        data/processed/normalization_log.csv          (every change / invalid value)

File naming convention (llm_raw): <model>_run1.csv, <model>_run2.csv, <model>_run3.csv,
<model>_shuffled.csv   e.g. claude_run1.csv, chatgpt_shuffled.csv
Human files: coder1.csv, coder2.csv
Invalid values are set to missing (never guessed) and logged.
"""
import pandas as pd

from common import (PROC_DIR, clean_abc, clean_bloom, list_rater_files,
                    rater_meta, read_coding_csv, read_original)


def main():
    orig = read_original()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    files = list_rater_files()
    if not files:
        raise SystemExit("No input files found.")

    long_parts, log = [], []
    for path, rtype in files:
        meta = rater_meta(path, rtype)
        rid = meta["rater_id"]
        df = read_coding_csv(path)
        for rep in df.attrs.get("repairs", []):
            log.append((rid, "", "file", "", "", "repair: " + rep))
        dups = df["key"].duplicated(keep="first")
        for k in df.loc[dups, "key"]:
            log.append((rid, k, "row", "", "", "duplicate row dropped (kept first)"))
        df = df[~dups].set_index("key")
        aligned = df.reindex(orig["key"])
        for k in aligned.index[aligned["bloom"].isna()]:
            log.append((rid, k, "row", "", "", "outcome missing in file"))

        bloom, abc = [], []
        for k, b, a in zip(aligned.index, aligned["bloom"].fillna(""), aligned["abc"].fillna("")):
            vb, sb = clean_bloom(b)
            va, sa = clean_abc(a)
            if k in df.index:
                if sb in ("normalized", "invalid", "empty"):
                    log.append((rid, k, "bloom", b, vb, sb))
                if sa in ("normalized", "invalid", "empty"):
                    log.append((rid, k, "abc", a, va, sa))
            bloom.append(vb)
            abc.append(va)

        part = orig[["stage", "grade", "code", "key"]].copy()
        part["rater_id"] = rid
        part["rater_type"] = meta["rater_type"]
        part["model"] = meta["model"]
        part["variant"] = meta["variant"]
        part["bloom"] = pd.array(bloom, dtype="Int64")
        part["abc"] = abc
        part["napomena"] = aligned["napomena"].fillna("").values
        long_parts.append(part)

    long = pd.concat(long_parts, ignore_index=True)
    long.drop(columns="key").to_csv(PROC_DIR / "all_codings_long.csv", index=False, encoding="utf-8")

    for col in ("bloom", "abc"):
        wide = long.pivot(index="key", columns="rater_id", values=col).reindex(orig["key"])
        wide.insert(0, "code", orig["code"].values)
        wide.insert(0, "grade", orig["grade"].values)
        wide.insert(0, "stage", orig["stage"].values)
        wide.to_csv(PROC_DIR / f"{col}_wide.csv", index=False, encoding="utf-8")

    pd.DataFrame(log, columns=["rater_id", "key", "field", "raw", "clean", "action"]).to_csv(
        PROC_DIR / "normalization_log.csv", index=False, encoding="utf-8")
    n_bad = sum(1 for r in log if r[5] != "normalized" and not r[5].startswith("repair:"))
    print(f"Merged {len(files)} raters x {len(orig)} outcomes. "
          f"{len(log)} log entries ({n_bad} problems, rest normalisations). "
          f"See data/processed/normalization_log.csv")


if __name__ == "__main__":
    main()