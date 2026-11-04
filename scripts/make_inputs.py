#!/usr/bin/env python3
"""Generate the shuffled input and ready-to-paste LLM prompts.

Reads : data/input/input_original.tsv, protocol/llm_prompt_en.txt
Writes: data/input/input_shuffled_seed<SEED>.tsv
        data/input/shuffle_order_seed<SEED>.csv
        protocol/prompts/prompt_original_order.txt
        protocol/prompts/prompt_shuffled_seed<SEED>.txt
"""
import argparse
import csv
import random

from common import INPUT_DIR, PROTOCOL_DIR, N_EXPECTED, make_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lines = (INPUT_DIR / "input_original.tsv").read_text(encoding="utf-8").splitlines()
    header, rows = lines[0].split("\t"), [l.split("\t") for l in lines[1:] if l.strip()]
    assert len(rows) == N_EXPECTED, f"expected {N_EXPECTED} rows, got {len(rows)}"
    keys = [make_key(r[0], r[2]) for r in rows]
    assert len(set(keys)) == len(keys), "duplicate stage+code keys"

    shuffled = rows[:]
    random.Random(args.seed).shuffle(shuffled)

    tsv = lambda rs: "\n".join("\t".join(r) for r in rs)
    hdr = "\t".join(header) + "\n"

    (INPUT_DIR / f"input_shuffled_seed{args.seed}.tsv").write_text(
        hdr + tsv(shuffled) + "\n", encoding="utf-8")

    pos = {k: i + 1 for i, k in enumerate(keys)}
    with open(INPUT_DIR / f"shuffle_order_seed{args.seed}.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shuffled_position", "original_position", "stage", "code"])
        for i, r in enumerate(shuffled, 1):
            w.writerow([i, pos[make_key(r[0], r[2])], r[0], r[2]])

    tmpl = (PROTOCOL_DIR / "llm_prompt_en.txt").read_text(encoding="utf-8")
    assert "{{INPUT}}" in tmpl
    out = PROTOCOL_DIR / "prompts"
    out.mkdir(exist_ok=True)
    (out / "prompt_original_order.txt").write_text(
        tmpl.replace("{{INPUT}}", tsv(rows)), encoding="utf-8")
    (out / f"prompt_shuffled_seed{args.seed}.txt").write_text(
        tmpl.replace("{{INPUT}}", tsv(shuffled)), encoding="utf-8")
    print(f"OK: {len(rows)} outcomes, seed {args.seed}. Files written.")


if __name__ == "__main__":
    main()
