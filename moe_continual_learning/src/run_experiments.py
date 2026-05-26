"""Main experiment driver.

Trains all model/strategy combinations sequentially on the SplitMNIST and
PermutedMNIST benchmarks and stores the accuracy matrices, the summary
metrics and the router profiles as JSON/NPZ files in ``../results``.

Usage:  python3 run_experiments.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import make_split_mnist, make_permuted_mnist
from models import BaselineMLP, WideMLP, MoENet, count_parameters
from continual import NaiveTrainer, EWCTrainer, MoETrainer, accuracy_matrix_at_end

RESULTS = "../results"
SEEDS = [0, 1, 2]
N_TASKS = 5
EPOCHS = 5
DEVICE = "cpu"


def build_trainers(seed):
    """Creates one trainer per compared approach (identical seed for all)."""
    torch.manual_seed(seed)
    naive = NaiveTrainer(BaselineMLP(), epochs=EPOCHS, device=DEVICE)
    torch.manual_seed(seed)
    ewc = EWCTrainer(BaselineMLP(), epochs=EPOCHS, device=DEVICE, ewc_lambda=500.0)
    torch.manual_seed(seed)
    wide = NaiveTrainer(WideMLP(), epochs=EPOCHS, device=DEVICE)
    wide.name = "Wide MLP (naive)"
    torch.manual_seed(seed)
    moe = MoETrainer(MoENet(n_experts=8, top_k=2), epochs=EPOCHS, device=DEVICE,
                     lb_coeff=0.01)
    return [naive, ewc, wide, moe]


def save_partial(results):
    """Writes the results collected so far to disk (crash-safe)."""
    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def run_benchmark(name, tasks_fn, results):
    """Runs all approaches on one benchmark for all seeds."""
    out = {"benchmark": name, "approaches": {}}
    for seed in SEEDS:
        tasks = tasks_fn(seed)
        for trainer in build_trainers(seed):
            t0 = time.time()
            accs = trainer.run(tasks)
            R, avg_acc, avg_forget, bwt = accuracy_matrix_at_end(accs)
            rec = {
                "seed": seed,
                "R": R.tolist(),
                "avg_acc": avg_acc,
                "avg_forgetting": avg_forget,
                "bwt": bwt,
                "params": count_parameters(trainer.model),
                "runtime_s": round(time.time() - t0, 1),
            }
            if hasattr(trainer, "router_profile"):
                profile = trainer.router_profile(tasks)
                rec["router_profile"] = profile.tolist()
                np.savez(os.path.join(
                    RESULTS, f"router_{name}_seed{seed}.npz"), profile=profile)
            out["approaches"].setdefault(trainer.name, []).append(rec)
            # Persist after every run so a crash cannot lose earlier runs.
            save_partial(results + [out])
            print(f"[{name}] seed={seed} {trainer.name}: "
                  f"avg_acc={avg_acc:.4f} forgetting={avg_forget:.4f} "
                  f"bwt={bwt:.4f} ({rec['runtime_s']}s)", flush=True)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    results = []

    print("=== SplitMNIST ===", flush=True)
    results.append(run_benchmark(
        "split_mnist",
        lambda seed: make_split_mnist(n_tasks=N_TASKS), results))

    print("=== PermutedMNIST ===", flush=True)
    results.append(run_benchmark(
        "permuted_mnist",
        lambda seed: make_permuted_mnist(n_tasks=N_TASKS, seed=seed), results))

    save_partial(results)
    print("Saved results to results/results.json", flush=True)


if __name__ == "__main__":
    main()
