"""Experiment: input-level MoE (MoEInputNet) on both benchmarks.

Motivation: the main experiment showed that hidden-level routing fails on
PermutedMNIST regardless of the load-balancing coefficient, because on this
benchmark the task-specific information lives in the input mapping, which is
stored in the *shared* trunk in front of the router.  MoEInputNet moves the
MoE layer to the input so that the router can also protect the input-side
mapping.

Usage:  python3 run_v2.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import make_split_mnist, make_permuted_mnist
from models import MoEInputNet, count_parameters
from continual import MoETrainer, accuracy_matrix_at_end

RESULTS = "../results"
SEEDS = [0, 1, 2]


def run(name, tasks):
    recs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        trainer = MoETrainer(MoEInputNet(), epochs=5, lb_coeff=0.01)
        trainer.name = "MoE (input-level)"
        t0 = time.time()
        accs = trainer.run(tasks(seed))
        R, avg_acc, avg_forget, bwt = accuracy_matrix_at_end(accs)
        profile = trainer.router_profile(tasks(seed))
        np.savez(os.path.join(RESULTS, f"router_v2_{name}_seed{seed}.npz"),
                 profile=profile)
        recs.append({
            "seed": seed, "R": R.tolist(), "avg_acc": avg_acc,
            "avg_forgetting": avg_forget, "bwt": bwt,
            "params": count_parameters(trainer.model),
            "runtime_s": round(time.time() - t0, 1),
            "router_profile": profile.tolist(),
        })
        print(f"[{name}] seed={seed} input-level MoE: avg_acc={avg_acc:.4f} "
              f"forgetting={avg_forget:.4f} bwt={bwt:.4f}", flush=True)
    return recs


def main():
    out = {"benchmarks": {}}
    out["benchmarks"]["split_mnist"] = run(
        "split_mnist", lambda seed: make_split_mnist(n_tasks=5))
    out["benchmarks"]["permuted_mnist"] = run(
        "permuted_mnist", lambda seed: make_permuted_mnist(n_tasks=5, seed=seed))
    with open(os.path.join(RESULTS, "results_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved results_v2.json", flush=True)


if __name__ == "__main__":
    main()
