"""
Quick diagnostic for the 'field' / 'Arable Land' orphan-root anomaly.

Run from anywhere once you have the label file downloaded (it's cached at
<image_root>/../UCM-HMLC.txt after the first check_ucm_temp.py run):

    python scripts/diagnose_hierarchy.py path/to/UCM-HMLC.txt
"""
import sys

import numpy as np


def main(label_file: str) -> None:
    with open(label_file, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    header = lines[0].split("\t")
    node_names = header[2:]

    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append([int(x) for x in parts[2:]])
    mat = np.array(rows)

    idx = {name: i for i, name in enumerate(node_names)}

    field_rows = set(np.nonzero(mat[:, idx["field"]])[0].tolist())
    agri_rows = set(np.nonzero(mat[:, idx["Agricultural Areas"]])[0].tolist())
    arable_rows = set(np.nonzero(mat[:, idx["Arable Land"]])[0].tolist())

    print(f"'field' positive rows       : {len(field_rows)}")
    print(f"'Agricultural Areas' rows   : {len(agri_rows)}")
    print(f"'Arable Land' rows          : {len(arable_rows)}")
    print(f"field rows NOT in Agri Areas: {len(field_rows - agri_rows)} "
          f"(should be 0 if field ⊆ Agricultural Areas)")
    print(f"field rows NOT in Arable    : {len(field_rows - arable_rows)} "
          f"(should be 0 if field ⊆ Arable Land)")
    print(f"Agri Areas rows NOT in field: {len(agri_rows - field_rows)} "
          f"(expected > 0 -- other leaves also roll up into Agricultural Areas)")

    if field_rows - agri_rows:
        example_rows = sorted(field_rows - agri_rows)[:5]
        print(f"\nExample row indices where field=1 but Agricultural Areas=0: {example_rows}")
        print("Full label vector for the first such row (all node_names):")
        r = example_rows[0]
        for name in node_names:
            val = mat[r, idx[name]]
            if val:
                print(f"  {name} = {val}")


if __name__ == "__main__":
    label_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/UCMerced_LandUse/UCM-HMLC.txt"
    main(label_file)
