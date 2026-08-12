# Continual Learning with Mixture of Experts (MoE)

Project work (Projektarbeit), M.Sc. Machine Learning and Data Analytics,
Hochschule Aalen.

**Author:** Gajera Yash Hasmukhbhai (Matriculation No. 3015550)

## Research Question

Can a Mixture of Experts (MoE) architecture learn a sequence of tasks
without catastrophically forgetting the earlier ones? The hypothesis: a
learned router sends different tasks to different subsets of experts, so
gradient updates on a new task mostly rewrite parameters that carry little
old knowledge.

## Repository Structure

```
moe_continual_learning/
├── data/                  # MNIST raw files (downloaded once)
├── src/
│   ├── datasets.py        # SplitMNIST and PermutedMNIST benchmark builders
│   ├── models.py          # BaselineMLP, WideMLP, MoENet, MoEInputNet
│   ├── continual.py       # Naive / EWC / MoE trainers, CL metrics
│   ├── run_experiments.py # main experiment driver (4 approaches, 2 benchmarks, 3 seeds)
│   ├── run_ablation.py    # ablation of the load-balancing coefficient
│   ├── run_v2.py          # input-level MoE variant on both benchmarks
│   ├── make_figures.py    # generates all report figures (PDF)
│   └── make_tables.py     # generates the LaTeX result tables
└── results/               # raw results (JSON) + router profiles (NPZ) + logs
```

## Benchmarks

| Benchmark      | Scenario           | Tasks            | Head protocol             |
|----------------|--------------------|------------------|---------------------------|
| SplitMNIST     | task-incremental   | 5 x 2 classes    | one 2-way head per task   |
| PermutedMNIST  | domain-incremental | 5 permutations   | single shared 10-way head |

## Compared Approaches

1. **Naive MLP**: 784-400-400 MLP, plain sequential fine tuning.
2. **MLP + EWC**: same network with Elastic Weight Consolidation (lambda = 500).
3. **Wide MLP (naive)**: 784-1100-1100 MLP, capacity control (2.07M params).
4. **MoE**: trunk + MoE layer (8 experts, top-2 gating, load balancing
   loss) + per-task heads (1.60M params).
5. **MoE (input-level)**: MoE layer directly on the input + shared hidden
   layer + per-task heads (1.68M params).

## Metrics

Accuracy matrix `R[i,j]` = test accuracy on task *j* after training task
*i*. From the final matrix: **average accuracy**, **average forgetting**,
and **backward transfer (BWT)** (Lopez-Paz and Ranzato, 2017).

## Reproducing the Experiments

Requirements: Python 3.12, PyTorch 2.8, torchvision, NumPy, Matplotlib.

```bash
cd src
python3 run_experiments.py   # main suite (saves results incrementally)
python3 run_ablation.py      # load-balancing ablation on PermutedMNIST
python3 run_v2.py            # input-level MoE on both benchmarks
python3 make_figures.py      # writes report figures as PDF
python3 make_tables.py       # writes LaTeX tables with the measured numbers
```

All random choices (initialisation, permutations, data order) are seeded
(seeds 0, 1, 2), so the runs are reproducible.

## Main Findings

* The naive baseline forgets severely on both benchmarks.
* On SplitMNIST the MoE model retains more than the naive baseline and the
  capacity matched wide baseline, so the benefit comes from the routing
  structure rather than from the parameter count.
* On PermutedMNIST the hidden level MoE fails: the shared trunk in front of
  the router stores the task specific input mapping and is overwritten by
  every new task, and the load balancing loss additionally forces all tasks
  onto all experts (dose dependent, see `results/ablation_lb.json`).
* Moving the MoE layer to the input (MoEInputNet) fixes this architectural
  mismatch and gives the best retention of all compared approaches on
  SplitMNIST.
* Router profiles (`results/router_*.npz`) show how different tasks are
  routed to different expert subsets, which is the mechanism behind the
  reduced forgetting.

See the project report (`report/main.pdf`) for the full analysis.
