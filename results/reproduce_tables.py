"""Regenerate the public manuscript tables from compact result inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_builders import TABLES, build_all, build_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Regenerate every public table")
    group.add_argument("--table", choices=sorted(TABLES), help="Regenerate one table")
    args = parser.parse_args()
    outputs = build_all() if args.all else [build_table(args.table)]
    for output in outputs:
        print(output.relative_to(Path.cwd()) if output.is_relative_to(Path.cwd()) else output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
