#!/usr/bin/env python3
"""Evaluate the model output.

Usage:
    evaluate.py <corrected> [--no-tokenize] --m2 <path_to_m2>
    evaluate.py (-h | --help)

Options:
    -h --help           Show this screen.

<corrected> is the path to the model output.

"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate the model output.")
    parser.add_argument("corrected", type=str, help="Path to the model output")
    parser.add_argument("--m2", type=str, help="Path to the golden annotated data (.m2 file)",
            required=True)
    args = parser.parse_args()
    tmp = Path(tempfile.gettempdir())

    tokenized_path = args.corrected

    # Get the source text out of m2
    source_path = tmp / f"unlp.source.tok"
    with open(args.m2) as f, open(source_path, "w") as out:
        for line in f:
            if line.startswith("S "):
                out.write(line[2:])

    # Align tokenized submission with the original text with Errant
    m2_target = tmp / "unlp.target.m2"
    subprocess.run(["errant_parallel", "-orig", source_path, "-cor", tokenized_path, "-out", m2_target], check=True)
    print(f"Aligned submission: {m2_target}", file=sys.stderr)

    # Evaluate
    subprocess.run(["errant_compare", "-hyp", m2_target, "-ref", args.m2])
    subprocess.run(["errant_compare", "-hyp", m2_target, "-ref", args.m2, "-ds", "-cat", "3"])


if __name__ == "__main__":
    main()
