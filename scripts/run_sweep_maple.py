import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def run_one(dataset, k, seed, epochs, checkpoint_dir, extra_args):
    cmd = [
        sys.executable, "scripts/train_maple.py",
        "--dataset", dataset,
        "--k", str(k),
        "--run_seed", str(seed),
        "--epochs", str(epochs),
        "--checkpoint_dir", checkpoint_dir,
    ] + extra_args
    print(f"\n{'='*70}\nRunning: dataset={dataset} k={k} seed={seed}\n{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: run failed (dataset={dataset}, k={k}, seed={seed}), "
              f"return code {result.returncode} -- continuing with remaining runs")
        return False
    return True


def aggregate(dataset, ks, checkpoint_dir):
    print(f"\n\n{'='*70}\nAGGREGATED RESULTS -- {dataset}\n{'='*70}")
    rows = []
    for k in ks:
        pattern = os.path.join(checkpoint_dir, f"results_{dataset}_k{k}_seed*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"K={k}: no results found (pattern: {pattern})")
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

        print(f"\nK={k} ({len(files)} run(s) found):")
        print(f"  Final  AUPRC: {fmt(final_auprcs)}   Ranking Loss: {fmt(final_rls)}")
        print(f"  Best   AUPRC: {fmt(best_auprcs)}   Ranking Loss: {fmt(best_rls)}")

        rows.append({
            "k": k, "n_runs": len(files),
            "final_auprc_mean": statistics.mean(final_auprcs) if final_auprcs else None,
            "final_auprc_std": statistics.stdev(final_auprcs) if len(final_auprcs) > 1 else None,
            "final_rl_mean": statistics.mean(final_rls) if final_rls else None,
            "final_rl_std": statistics.stdev(final_rls) if len(final_rls) > 1 else None,
            "best_auprc_mean": statistics.mean(best_auprcs) if best_auprcs else None,
            "best_auprc_std": statistics.stdev(best_auprcs) if len(best_auprcs) > 1 else None,
        })

    csv_path = os.path.join(checkpoint_dir, f"sweep_summary_{dataset}.csv")
    with open(csv_path, "w") as f:
        f.write("k,n_runs,final_auprc_mean,final_auprc_std,final_rl_mean,final_rl_std,"
                "best_auprc_mean,best_auprc_std\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in [
                "k", "n_runs", "final_auprc_mean", "final_auprc_std",
                "final_rl_mean", "final_rl_std", "best_auprc_mean", "best_auprc_std",
            ]) + "\n")
    print(f"\nSaved summary CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ucm", "aid"])
    parser.add_argument("--ks", type=str, default="4,8,12,16")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/maple_checkpoints")
    parser.add_argument("--force_rerun", action="store_true",
                         help="rerun even if a results JSON already exists for that (k, seed)")
    parser.add_argument("--aggregate_only", action="store_true",
                         help="skip training entirely, just aggregate existing results")
    args, extra_args = parser.parse_known_args()

    ks = [int(x) for x in args.ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not args.aggregate_only:
        total = len(ks) * len(seeds)
        done = 0
        for k in ks:
            for seed in seeds:
                done += 1
                results_path = os.path.join(
                    args.checkpoint_dir, f"results_{args.dataset}_k{k}_seed{seed}.json"
                )
                if os.path.exists(results_path) and not args.force_rerun:
                    print(f"[{done}/{total}] SKIPPING dataset={args.dataset} k={k} seed={seed} "
                          f"-- results already exist at {results_path} (use --force_rerun to redo)")
                    continue
                print(f"[{done}/{total}] Starting dataset={args.dataset} k={k} seed={seed}")
                run_one(args.dataset, k, seed, args.epochs, args.checkpoint_dir, extra_args)

    aggregate(args.dataset, ks, args.checkpoint_dir)


if __name__ == "__main__":
    main()