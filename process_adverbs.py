#!/usr/bin/env python3
"""
Split `src/yaml/adv.all.yaml` into multiple files by supersense/type using
the mapping in `adverb_wordnet_llm.csv`.

Behavior:
- Reads YAML file that is expected to be a top-level mapping keyed by synset id
  strings (e.g. "00001885-r").
- Reads CSV file with at least the columns: "synset" and "type".
- Builds a mapping synset -> type, keeping only the first type seen for a synset.
- Groups synsets from the YAML by type and writes one YAML file per type:
    src/yaml/adv.<slug(type)>.yaml
  and an `src/yaml/adv.UNMAPPED.yaml` file for synsets that have no mapping.
- By default the script runs in dry-run mode printing counts and samples.
  Use `--write` to actually write files. Use `--backup` to back up the original
  YAML before writing.

Example:
  python3 process_adverbs.py        # dry-run, print summary
  python3 process_adverbs.py --write --backup
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from typing import Dict, List, Tuple

try:
    import yaml
except Exception as e:
    print(
        "Error: PyYAML is required to run this script. Install with `pip install pyyaml`.",
        file=sys.stderr,
    )
    raise

DEFAULT_YAML = os.path.join("src", "yaml", "adv.all.yaml")
DEFAULT_CSV = "adverb_wordnet_llm.csv"
OUT_DIR_DEFAULT = os.path.join("src", "yaml")
PREFIX = "adv"
UNMAPPED_NAME = "UNMAPPED"


def slugify(s: str) -> str:
    """Convert a type string into a filesystem-friendly slug."""
    s = (s or "").strip().lower()
    # replace non-alnum with underscore
    s = s.split(".")[-1]
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "unknown"


def read_yaml(path: str) -> Dict[str, object]:
    """Load YAML and expect a top-level mapping of synset id -> data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected YAML structure: expected mapping at top-level in {path}"
        )
    return data


def read_csv_mapping(path: str) -> Dict[str, str]:
    """
    Read CSV and return a mapping synset_id -> type.
    Keep the first type seen for a synset (user requested).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    mapping: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "synset" not in headers or "type" not in headers:
            raise ValueError(f"CSV missing required columns. Found headers: {headers}")
        for row in reader:
            syn = (row.get("synset") or "").strip()
            t = (row.get("type") or "").strip()
            if not syn:
                continue
            # Keep first type only
            if syn not in mapping and t:
                mapping[syn] = t
    return mapping


def group_synsets(
    yaml_data: Dict[str, object], csv_map: Dict[str, str]
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object], List[str]]:
    """
    Return:
      - groups: dict mapping type -> dict of synset_id -> synset_data
      - unmapped: dict of synset_id -> synset_data for those with no CSV mapping
      - csv_only: list of synset ids that appear in CSV but not in YAML
    """
    groups: Dict[str, Dict[str, object]] = {}
    unmapped: Dict[str, object] = {}

    for sid, syn in yaml_data.items():
        t = csv_map.get(sid)
        if t:
            groups.setdefault(t, {})[sid] = syn
        else:
            unmapped[sid] = syn

    csv_only = [s for s in csv_map.keys() if s not in yaml_data]
    return groups, unmapped, csv_only


def write_yaml_file(path: str, data: Dict[str, object]) -> None:
    """Write data (a mapping) to YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        # sort_keys=False tries to preserve insertion order for dicts
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_unmapped_csv(path: str, unmapped: Dict[str, object]) -> None:
    """
    Write unmapped entries to a CSV file matching the structure of
    adverb_wordnet_llm.csv: columns = synset,adverb,sentence,definition,type.

    For each synset in `unmapped`, write one row per member (adverb lemma).
    Use the first example (if present) as the sentence; include the definition
    (joined if multiple) in the `definition` column; leave `type` empty.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["synset", "adverb", "sentence", "definition", "type"])
        # iterate in deterministic order
        for sid, syn in sorted(unmapped.items()):
            # members are usually under 'members' in the YAML
            members = []
            mval = syn.get("members") if isinstance(syn, dict) else None
            if isinstance(mval, list):
                members = mval
            else:
                # fallback to other possible keys
                mval2 = syn.get("lemmas") if isinstance(syn, dict) else None
                if isinstance(mval2, list):
                    members = mval2

            # pick first example sentence if available
            sentence = syn.get("example") if isinstance(syn, dict) else None
            if sentence is not None:
                sentence = sentence[0]

            # extract definition: join list entries if necessary
            definition = ""
            defval = syn.get("definition") if isinstance(syn, dict) else None
            if isinstance(defval, list):
                # join multiple definition strings with " | "
                definition = " | ".join(str(d).strip() for d in defval if d is not None)
            elif isinstance(defval, str):
                definition = defval.strip()

            # If there are no explicit members, still emit one row with empty adverb
            if not members:
                writer.writerow([sid, "", sentence, definition, ""])
            else:
                for m in members:
                    writer.writerow([sid, m, sentence, definition, ""])


def write_groups(
    groups: Dict[str, Dict[str, object]],
    unmapped: Dict[str, object],
    out_dir: str,
    prefix: str = PREFIX,
    overwrite: bool = False,
) -> List[str]:
    """
    Write group files and the unmapped file.
    Returns list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    for t, syns in groups.items():
        slug = slugify(t)
        filename = f"{prefix}.{slug}.yaml"
        out_path = os.path.join(out_dir, filename)
        if os.path.exists(out_path) and not overwrite:
            print(
                f"Skipping existing file {out_path} (use --overwrite to overwrite)",
                file=sys.stderr,
            )
            continue
        write_yaml_file(out_path, syns)
        written.append(out_path)

    # Write unmapped entries as CSV if any (follow adverb_wordnet_llm.csv schema)
    unmapped_path = os.path.join(out_dir, f"{prefix}.{UNMAPPED_NAME}.yaml")
    unmapped_csv = os.path.join(out_dir, f"{prefix}.{UNMAPPED_NAME}.csv")
    if unmapped:
        write_yaml_file(unmapped_path, unmapped)
        write_unmapped_csv(unmapped_csv, unmapped)
        written.append(unmapped_csv)

    return written


def print_summary(
    yaml_data: Dict[str, object],
    groups: Dict[str, Dict[str, object]],
    unmapped: Dict[str, object],
    csv_only: List[str],
    sample_size: int = 10,
) -> None:
    total = len(yaml_data)
    print(f"Total synsets in YAML: {total}")
    print("Groups (type -> count):")
    # Sort by descending size then name
    for t, syns in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"  {t}: {len(syns)}")
    print(f"Unmapped synsets: {len(unmapped)}")
    if csv_only:
        print(f"CSV-only synsets (in CSV but not in YAML): {len(csv_only)}")
        print(f"  sample: {csv_only[:sample_size]}")
    # show small samples from a few groups
    print("\nSample mappings (type -> up to 5 synset ids):")
    shown = 0
    for t, syns in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sample = list(syns.keys())[:5]
        print(f"  {t}: {sample}")
        shown += 1
        if shown >= 8:
            break


def backup_file(path: str) -> str:
    """Make a backup copy of path next to it with .bak suffix. Return backup path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot backup non-existent file: {path}")
    bak = path + ".bak"
    shutil.copy2(path, bak)
    return bak


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split adv.all.yaml into per-type files using adverb_wordnet_llm.csv"
    )
    p.add_argument(
        "--yaml",
        default=DEFAULT_YAML,
        help="path to adv.all.yaml (default: src/yaml/adv.all.yaml)",
    )
    p.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="path to CSV mapping (default: adverb_wordnet_llm.csv)",
    )
    p.add_argument(
        "--out-dir",
        default=OUT_DIR_DEFAULT,
        help="directory where output files will be written (default: src/yaml)",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="actually write output files (default: dry-run)",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="when writing, backup original YAML to <yaml>.bak",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="when writing, overwrite any existing target files",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=10,
        help="number of samples to show in dry-run summary",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        yaml_data = read_yaml(args.yaml)
    except Exception as e:
        print(f"Error reading YAML: {e}", file=sys.stderr)
        return 2

    try:
        csv_map = read_csv_mapping(args.csv)
    except Exception as e:
        print(f"Error reading CSV mapping: {e}", file=sys.stderr)
        return 3

    groups, unmapped, csv_only = group_synsets(yaml_data, csv_map)
    print_summary(yaml_data, groups, unmapped, csv_only, sample_size=args.sample)

    if not args.write and not args.overwrite:
        print(
            "\nDry-run complete. No files written. Re-run with --write to produce output files."
        )
        return 0

    # writing mode
    if args.backup:
        try:
            bak = backup_file(args.yaml)
            print(f"Backed up YAML to: {bak}")
        except Exception as e:
            print(f"Could not create backup: {e}", file=sys.stderr)
            return 4

    try:
        written = write_groups(
            groups, unmapped, args.out_dir, prefix=PREFIX, overwrite=args.overwrite
        )
    except Exception as e:
        print(f"Error writing files: {e}", file=sys.stderr)
        return 5

    if written:
        print("\nWrote files:")
        for p in written:
            print("  " + p)
    else:
        print(
            "\nNo files were written (all target files already exist and --overwrite was not set)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
