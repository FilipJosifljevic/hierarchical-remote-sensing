import argparse
import glob
import json
import os
import statistics
import subprocess
import sys


def run_one(fraction, seed, checkpoint_dir, extra_args):
    cmd = [
        sys.executable, "scripts/train_helm.py",
        "--labeled_fraction", str(fraction),
        "--run_seed", str(seed),
        "--checkpoint_dir", checkpoint_dir,
    ] + extra_args
    print(f"\n{'='*70}\nRunning: fraction={fraction} seed={seed}\n{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: run failed (fraction={fraction}, seed={seed}), "
              f"return code {result.returncode} -- continuing with remaining runs")
        return False
    return True


def aggregate(fractions, checkpoint_dir, run_tag):
    tag_suffix = f"_{run_tag}" if run_tag else ""
    print(f"\n\n{'='*70}\nAGGREGATED RESULTS{f' (tag={run_tag})' if run_tag else ''}\n{'='*70}")
    rows = []
    for frac in fractions:
        pattern = os.path.join(checkpoint_dir, f"results_frac{frac}_seed*{tag_suffix}.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"fraction={frac}: no results found (pattern: {pattern})")
            continue

        final_auprcs, final_rls, best_auprcs, best_rls = [], [], [], []
        for f in files:
            with open(f) as fh:
                r = json.load(fh)
            if r["final_metrics"]:
                final_auprcs.append(r["final_metrics"]["auprc"])
                final_rls.append(r["final_metrics"]["ranking_loss"])
            if r["best_metrics"]:
                best_auprcs.append(r["best_metrics"]["auprc"])
                best_rls.append(r["best_metrics"]["ranking_loss"])

        def fmt(values):
            if len(values) == 0:
                return "n/a"
            if len(values) == 1:
                return f"{values[0]:.4f} (n=1)"
            return f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f} (n={len(values)})"

        print(f"\nfraction={frac} ({len(files)} run(s) found):")
        print(f"  Final  AUPRC: {fmt(final_auprcs)}   Ranking Loss: {fmt(final_rls)}")
        print(f"  Best   AUPRC: {fmt(best_auprcs)}   Ranking Loss: {fmt(best_rls)}")

        rows.append({
            "fraction": frac, "n_runs": len(files),
            "final_auprc_mean": statistics.mean(final_auprcs) if final_auprcs else None,
            "final_auprc_std": statistics.stdev(final_auprcs) if len(final_auprcs) > 1 else None,
            "best_auprc_mean": statistics.mean(best_auprcs) if best_auprcs else None,
            "best_auprc_std": statistics.stdev(best_auprcs) if len(best_auprcs) > 1 else None,
        })

    csv_path = os.path.join(checkpoint_dir, f"sweep_summary_helm{tag_suffix}.csv")
    with open(csv_path, "w") as f:
        f.write("fraction,n_runs,final_auprc_mean,final_auprc_std,best_auprc_mean,best_auprc_std\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in [
                "fraction", "n_runs", "final_auprc_mean", "final_auprc_std",
                "best_auprc_mean", "best_auprc_std",
            ]) + "\n")
    print(f"\nSaved summary CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=str, default="0.01,0.05,0.10,0.25")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--run_tag", type=str, default=None,
                         help="passed through to train.py -- keeps this sweep's files "
                              "separate from other runs")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args, extra_args = parser.parse_known_args()

    if args.run_tag:
        extra_args += ["--run_tag", args.run_tag]

    fractions = [float(x) for x in args.fractions.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not args.aggregate_only:
        total = len(fractions) * len(seeds)
        done = 0
        for frac in fractions:
            for seed in seeds:
                done += 1
                tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
                results_path = os.path.join(
                    args.checkpoint_dir, f"results_frac{frac}_seed{seed}{tag_suffix}.json"
                )
                if os.path.exists(results_path) and not args.force_rerun:
                    print(f"[{done}/{total}] SKIPPING fraction={frac} seed={seed} "
                          f"-- results already exist (use --force_rerun to redo)")
                    continue
                print(f"[{done}/{total}] Starting fraction={frac} seed={seed}")
                run_one(frac, seed, args.checkpoint_dir, extra_args)

    aggregate(fractions, args.checkpoint_dir, args.run_tag)


if __name__ == "__main__":
    main()