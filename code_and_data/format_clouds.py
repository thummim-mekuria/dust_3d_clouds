#!/usr/bin/env python3
"""Restore the aligned, space-padded formatting of the Zucker (2020) cloud table.

Given a whitespace-stripped CSV (bare values, no padding), this rewrites it in
the original column-aligned layout: the cloud name is left-justified and every
numeric column is right-justified to a fixed per-column width.

Usage:
    python3 format_clouds.py zucker2020_clouds_within_1200pc.csv
    python3 format_clouds.py input_stripped.csv output_aligned.csv
"""

import sys

# Exact header line from the original file (its d_mean/d_std labels carry a
# leading space, so it is stored verbatim rather than derived).
HEADER = (
    "cloud           ,l_min(deg),l_max(deg),b_min(deg),b_max(deg),"
    " d_mean(pc), d_std(pc),n_sightlines"
)

# Per-column layout for data rows: (width, justify). "<" = left, ">" = right.
COLSPECS = [
    (16, "<"),  # cloud
    (10, ">"),  # l_min(deg)
    (10, ">"),  # l_max(deg)
    (10, ">"),  # b_min(deg)
    (10, ">"),  # b_max(deg)
    (9, ">"),   # d_mean(pc)
    (7, ">"),   # d_std(pc)
    (5, ">"),   # n_sightlines
]


def format_row(cells):
    out = []
    for cell, (width, justify) in zip(cells, COLSPECS):
        out.append("{0:{1}{2}}".format(cell, justify, width))
    return ",".join(out)


def align(text):
    lines = text.splitlines()
    rows = [ln.split(",") for ln in lines if ln.strip()]
    # Drop the incoming header (row 0) and re-emit the canonical one.
    body = [format_row([c.strip() for c in r]) for r in rows[1:]]
    return "\n".join([HEADER, *body]) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    with open(src) as f:
        result = align(f.read())
    with open(dst, "w") as f:
        f.write(result)
    print("Aligned {0} -> {1}".format(src, dst))


if __name__ == "__main__":
    main()
