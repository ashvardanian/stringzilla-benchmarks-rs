#!/usr/bin/env python3
"""Extract unique already-lowercase terms from a whitespace-delimited frequency dictionary."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    seen: set[str] = set()
    with open(args.input, encoding="utf-8") as source, open(args.output, "w", encoding="utf-8") as target:
        for line in source:
            fields = line.split()
            if not fields:
                continue
            term = fields[0]
            if term.lower() != term or term in seen:
                continue
            seen.add(term)
            target.write(term + "\n")


if __name__ == "__main__":
    main()
