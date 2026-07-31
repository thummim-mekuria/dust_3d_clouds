#!/usr/bin/env python3
"""Strip all padding whitespace from a CSV, trimming every field.

Usage:
    python3 strip_clouds.py zucker2020_clouds_within_1200pc.csv
    python3 strip_clouds.py input.csv output.csv
"""

import sys


def strip(text):
    lines = text.splitlines()
    out = [
        ",".join(cell.strip() for cell in ln.split(","))
        for ln in lines
        if ln.strip()
    ]
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    with open(src) as f:
        result = strip(f.read())
    with open(dst, "w") as f:
        f.write(result)
    print("Stripped {0} -> {1}".format(src, dst))


if __name__ == "__main__":
    main()
