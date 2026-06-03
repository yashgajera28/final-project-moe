"""Generates all report figures from the stored experiment results.

Reads ``../results/results.json`` and the router profile NPZ files and
writes publication-quality PDF figures to ``../../report/figures``.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "../results"
FIGDIR = "../../report/figures"
APPROACHES = ["Naive MLP", "MLP + EWC", "Wide MLP (naive)", "MoE"]
COLORS = {"Naive MLP": "#c0392b", "MLP + EWC": "#2980b9",
          "Wide MLP (naive)": "#7f8c8d", "MoE": "#e67e22"}
BENCH_TITLES = {"split_mnist": "SplitMNIST", "permuted_mnist": "PermutedMNIST"}


def load():
    with open(os.path.join(RESULTS, "results.json")) as f:
        return {r["benchmark"]: r["approaches"] for r in json.load(f)}


def mean_R(recs):
    return np.nanmean(np.array([r["R"] for r in recs]), axis=0)


def fig_accuracy_heatmaps(data):
    """One heatmap of the mean accuracy matrix per benchmark and approach."""
    for bench, apps in data.items():
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
        for ax, name in zip(axes, APPROACHES):
            R = mean_R(apps[name])
            im = ax.imshow(R, vmin=0.4, vmax=1.0, cmap="viridis")
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    if not np.isnan(R[i, j]):
                        ax.text(j, i, f"{R[i, j]:.2f}", ha="center",
                                va="center", fontsize=7,
                                color="white" if R[i, j] < 0.85 else "black")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("task")
            ax.set_xticks(range(5), [f"T{i+1}" for i in range(5)])
            ax.set_yticks(range(5), [f"T{i+1}" for i in range(5)])
        axes[0].set_ylabel("after training stage")
        fig.suptitle(f"{BENCH_TITLES[bench]}: mean accuracy matrix $R$ "
                     f"(3 seeds)", fontsize=11)
        fig.colorbar(im, ax=axes, shrink=0.8, label="accuracy")
        fig.tight_layout(rect=[0, 0, 0.93, 0.95])
        fig.savefig(os.path.join(FIGDIR, f"accmatrix_{bench}.pdf"))
        plt.close(fig)


def fig_accuracy_curves(data):
    """Accuracy on every task as a function of the training stage."""
    for bench, apps in data.items():
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
        for ax, name in zip(axes, APPROACHES):
            R = mean_R(apps[name])
            for j in range(5):
                xs = np.arange(j, 5)
                ys = R[j:, j]
                ax.plot(xs + 1, ys, marker="o", ms=3, lw=1.4,
                        label=f"task {j+1}")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("training stage")
            ax.set_xticks(range(1, 6))
            ax.set_ylim(0.3, 1.02)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("test accuracy")
        axes[-1].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{BENCH_TITLES[bench]}: per-task accuracy over the "
                     f"task sequence (mean of 3 seeds)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(FIGDIR, f"curves_{bench}.pdf"))
        plt.close(fig)


def fig_metrics_summary(data):
    """Grouped bars: avg accuracy and forgetting on both benchmarks."""
    benches = list(data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, title in zip(
            axes, ["avg_acc", "avg_forgetting"],
            ["Average accuracy (higher is better)",
             "Average forgetting (lower is better)"]):
        x = np.arange(len(APPROACHES))
        width = 0.35
        for k, bench in enumerate(benches):
            means = [np.mean([r[metric] for r in data[bench][a]])
                     for a in APPROACHES]
            stds = [np.std([r[metric] for r in data[bench][a]])
                    for a in APPROACHES]
            bars = ax.bar(x + (k - 0.5) * width, means, width,
                          yerr=stds, capsize=3,
                          label=BENCH_TITLES[bench], alpha=0.9)
            for b, m in zip(bars, means):
                ax.text(b.get_x() + b.get_width() / 2, m + 0.012,
                        f"{m:.3f}", ha="center", fontsize=6.5)
        ax.set_xticks(x, [a.replace(" (naive)", "\n(naive)").replace(" + ", "\n+ ")
                          for a in APPROACHES], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.0 if metric == "avg_acc" else 0.35)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "metrics_summary.pdf"))
    plt.close(fig)


def fig_router_profiles():
    """Heatmap of the mean router gate probabilities per task."""
    for bench in ["split_mnist", "permuted_mnist"]:
        profiles = []
        for seed in [0, 1, 2]:
            path = os.path.join(RESULTS, f"router_{bench}_seed{seed}.npz")
            profiles.append(np.load(path)["profile"])
        P = np.mean(profiles, axis=0)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        im = ax.imshow(P, cmap="magma", vmin=0, vmax=P.max())
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if P[i, j] < 0.6 * P.max() else "black")
        ax.set_xticks(range(8), [f"E{i+1}" for i in range(8)])
        ax.set_yticks(range(5), [f"task {i+1}" for i in range(5)])
        ax.set_xlabel("expert")
        ax.set_ylabel("task")
        ax.set_title(f"{BENCH_TITLES[bench]}: mean gate probability per task "
                     f"and expert (3 seeds)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label="mean gate probability")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"router_{bench}.pdf"))
        plt.close(fig)


def fig_router_overlap():
    """Pairwise task overlap of the routing distributions."""
    for bench in ["split_mnist", "permuted_mnist"]:
        profiles = []
        for seed in [0, 1, 2]:
            path = os.path.join(RESULTS, f"router_{bench}_seed{seed}.npz")
            profiles.append(np.load(path)["profile"])
        P = np.mean(profiles, axis=0)
        Pn = P / P.sum(axis=1, keepdims=True)
        # cosine similarity between per-task routing distributions
        norm = Pn / np.linalg.norm(Pn, axis=1, keepdims=True)
        S = norm @ norm.T
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(S, cmap="cividis", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if S[i, j] < 0.6 else "black")
        ax.set_xticks(range(5), [f"T{i+1}" for i in range(5)])
        ax.set_yticks(range(5), [f"T{i+1}" for i in range(5)])
        ax.set_xlabel("task")
        ax.set_ylabel("task")
        ax.set_title(f"{BENCH_TITLES[bench]}: routing similarity\n"
                     f"between tasks (cosine)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"overlap_{bench}.pdf"))
        plt.close(fig)


def fig_router_v2():
    """Router profiles of the input-level MoE on both benchmarks."""
    for bench in ["split_mnist", "permuted_mnist"]:
        paths = [os.path.join(RESULTS, f"router_v2_{bench}_seed{s}.npz")
                 for s in [0, 1, 2]]
        if not all(os.path.exists(p) for p in paths):
            return
        P = np.mean([np.load(p)["profile"] for p in paths], axis=0)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        im = ax.imshow(P, cmap="magma", vmin=0, vmax=P.max())
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if P[i, j] < 0.6 * P.max() else "black")
        ax.set_xticks(range(8), [f"E{i+1}" for i in range(8)])
        ax.set_yticks(range(5), [f"task {i+1}" for i in range(5)])
        ax.set_xlabel("expert")
        ax.set_ylabel("task")
        ax.set_title(f"{BENCH_TITLES[bench]}: input-level router profile "
                     f"(3 seeds)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label="mean gate probability")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"router_v2_{bench}.pdf"))
        plt.close(fig)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    os.makedirs(FIGDIR, exist_ok=True)
    data = load()
    fig_accuracy_heatmaps(data)
    fig_accuracy_curves(data)
    fig_metrics_summary(data)
    fig_router_profiles()
    fig_router_overlap()
    fig_router_v2()
    print("figures written to", FIGDIR)
