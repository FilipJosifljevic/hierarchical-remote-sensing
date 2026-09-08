import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                         help="defaults to {checkpoint_dir}/all_results_merged.json")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.checkpoint_dir, "all_results_merged.json")

    pattern = os.path.join(args.checkpoint_dir, "results_*.json")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} results files matching {pattern}")

    all_results = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        r["_source_file"] = os.path.basename(f)
        all_results.append(r)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Merged {len(all_results)} runs into {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()