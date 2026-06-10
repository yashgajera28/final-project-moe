"""Ablation: effect of the load-balancing coefficient on PermutedMNIST.

The main experiment revealed that the MoE model with the standard
load-balancing coefficient (alpha = 0.01) forgets heavily on PermutedMNIST.
This script retrains the MoE model with weaker coefficients to test the
hypothesis that strong load balancing forces all tasks onto all experts and
thereby destroys task specialisation.

Usage:  python3 run_ablation.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import make_permuted_mnist
from models import MoENet, count_parameters
from continual import MoETrainer, accuracy_matrix_at_end

RESULTS = "../results"
SEEDS = [0, 1, 2]
LB_COEFFS = [0.001, 0.0]


def main():
    out = {"benchmark": "permuted_mnist", "approaches": {}}
    for lb in LB_COEFFS:
        name = f"MoE (lb={lb})"
        for seed in SEEDS:
            tasks = make_permuted_mnist(n_tasks=5, seed=seed)
            torch.manual_seed(seed)
            trainer = MoETrainer(MoENet(n_experts=8, top_k=2),
                                 epochs=5, lb_coeff=lb)
            trainer.name = name
            t0 = time.time()
            accs = trainer.run(tasks)
            R, avg_acc, avg_forget, bwt = accuracy_matrix_at_end(accs)
            rec = {
                "seed": seed, "R": R.tolist(), "avg_acc": avg_acc,
                "avg_forgetting": avg_forget, "bwt": bwt,
                "params": count_parameters(trainer.model),
                "runtime_s": round(time.time() - t0, 1),
            }
            profile = trainer.router_profile(tasks)
            rec["router_profile"] = profile.tolist()
            np.savez(os.path.join(
                RESULTS, f"router_permuted_mnist_lb{lb}_seed{seed}.npz"),
                profile=profile)
            out["approaches"].setdefault(name, []).append(rec)
            print(f"lb={lb} seed={seed}: avg_acc={avg_acc:.4f} "
                  f"forgetting={avg_forget:.4f} bwt={bwt:.4f} "
                  f"({rec['runtime_s']}s)", flush=True)

    with open(os.path.join(RESULTS, "ablation_lb.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved ablation_lb.json", flush=True)


if __name__ == "__main__":
    main()
