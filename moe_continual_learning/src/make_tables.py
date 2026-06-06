"""Generates LaTeX result tables from the stored experiment results.

Writes ``../../report/chapters/07_tables.tex`` so that every number in the
report is produced directly from the measured data (no manual copying).
"""

import json
import os

import numpy as np

RESULTS = "../results"
OUTDIR = "../../report/chapters"
APPROACHES = ["Naive MLP", "MLP + EWC", "Wide MLP (naive)", "MoE"]
BENCH_TITLES = {"split_mnist": "SplitMNIST", "permuted_mnist": "PermutedMNIST"}


def fmt(mean, std, best=False):
    s = f"${mean:.4f} \\pm {std:.4f}$"
    return f"\\textbf{{{s}}}" if best else s


def stats(recs, key):
    vals = [r[key] for r in recs]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    with open(os.path.join(RESULTS, "results.json")) as f:
        data = {r["benchmark"]: r["approaches"] for r in json.load(f)}

    # optional ablation results
    abl = None
    abl_path = os.path.join(RESULTS, "ablation_lb.json")
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            abl = json.load(f)["approaches"]

    out = {}
    for bench, apps in data.items():
        lines = []
        title = BENCH_TITLES[bench]
        lines.append(
            "\\begin{table}[H]\n  \\centering\n  \\small\n"
            f"  \\caption{{Results on {title} (mean $\\pm$ standard deviation "
            "over three seeds; best value per column in bold).}\n"
            f"  \\label{{tab:results_{bench}}}\n"
            "  \\begin{tabular}{lccc}\n    \\toprule\n"
            "    \\textbf{Approach} & \\textbf{Avg.\\ accuracy} $\\uparrow$ "
            "& \\textbf{Avg.\\ forgetting} $\\downarrow$ "
            "& \\textbf{BWT} $\\uparrow$ \\\\\n    \\midrule")

        # determine best values per column
        best_acc, best_forget, best_bwt = -1, 1, -1
        for a in APPROACHES:
            m_acc, _ = stats(apps[a], "avg_acc")
            m_for, _ = stats(apps[a], "avg_forgetting")
            m_bwt, _ = stats(apps[a], "bwt")
            best_acc = max(best_acc, m_acc)
            best_forget = min(best_forget, m_for)
            best_bwt = max(best_bwt, m_bwt)

        for a in APPROACHES:
            m_acc, s_acc = stats(apps[a], "avg_acc")
            m_for, s_for = stats(apps[a], "avg_forgetting")
            m_bwt, s_bwt = stats(apps[a], "bwt")
            lines.append(
                f"    {a} & {fmt(m_acc, s_acc, abs(m_acc-best_acc)<1e-9)} & "
                f"{fmt(m_for, s_for, abs(m_for-best_forget)<1e-9)} & "
                f"{fmt(m_bwt, s_bwt, abs(m_bwt-best_bwt)<1e-9)} \\\\")
        lines.append("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
        out[f"07_table_{bench}.tex"] = lines

    # per-task final accuracies
    lines = []
    lines.append(
        "\\begin{table}[H]\n  \\centering\n  \\small\n"
        "  \\caption{Final test accuracy per task after the complete task "
        "sequence (mean over three seeds).}\n"
        "  \\label{tab:pertask}\n"
        "  \\begin{tabular}{llccccc}\n    \\toprule\n"
        "    \\textbf{Benchmark} & \\textbf{Approach} "
        "& \\textbf{T1} & \\textbf{T2} & \\textbf{T3} & \\textbf{T4} "
        "& \\textbf{T5} \\\\\n    \\midrule")
    for bench, apps in data.items():
        for k, a in enumerate(APPROACHES):
            finals = np.array([[r["R"][4][j] for j in range(5)]
                               for r in apps[a]])
            means = finals.mean(axis=0)
            cells = " & ".join(f"{v:.4f}" for v in means)
            prefix = (f"    \\multirow{{4}}{{*}}{{{BENCH_TITLES[bench]}}} & "
                      if k == 0 else "    & ")
            lines.append(f"{prefix}{a} & {cells} \\\\")
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines.append("  \\end{tabular}\n\\end{table}\n")
    out["07_table_pertask.tex"] = lines

    # input-level MoE variant table
    v2_path = os.path.join(RESULTS, "results_v2.json")
    if os.path.exists(v2_path):
        with open(v2_path) as f:
            v2 = json.load(f)["benchmarks"]
        v2_lines = []
        v2_lines.append(
            "\\begin{table}[H]\n  \\centering\n  \\small\n"
            "  \\caption{Input-level MoE compared to all approaches on both "
            "benchmarks (mean $\\pm$ std over three seeds).}\n"
            "  \\label{tab:v2}\n"
            "  \\resizebox{\\textwidth}{!}{%\n"
            "  \\begin{tabular}{llccc}\n    \\toprule\n"
            "    \\textbf{Benchmark} & \\textbf{Approach} & \\textbf{Avg.\\ "
            "accuracy} $\\uparrow$ & \\textbf{Avg.\\ forgetting} $\\downarrow$ "
            "& \\textbf{BWT} $\\uparrow$ \\\\\n    \\midrule")
        for bench in ["permuted_mnist", "split_mnist"]:
            rows = [(a, data[bench][a]) for a in APPROACHES]
            rows.append(("MoE (input-level)", v2[bench]))
            for k, (label, recs) in enumerate(rows):
                m_acc, s_acc = stats(recs, "avg_acc")
                m_for, s_for = stats(recs, "avg_forgetting")
                m_bwt, s_bwt = stats(recs, "bwt")
                prefix = (f"    \\multirow{{5}}{{*}}{{{BENCH_TITLES[bench]}}} & "
                          if k == 0 else "    & ")
                v2_lines.append(
                    f"{prefix}{label} & {fmt(m_acc, s_acc)} & "
                    f"{fmt(m_for, s_for)} & {fmt(m_bwt, s_bwt)} \\\\")
            v2_lines.append("    \\midrule")
        v2_lines[-1] = "    \\bottomrule"
        v2_lines.append("  \\end{tabular}}%\n\\end{table}\n")
        out["07_table_v2.tex"] = v2_lines

    # ablation table
    if abl:
        abl_lines = []
        abl_lines.append(
            "\\begin{table}[H]\n  \\centering\n  \\small\n"
            "  \\caption{Ablation of the load-balancing coefficient "
            "$\\alpha$ on PermutedMNIST (mean $\\pm$ std over three seeds)."
            "}\n  \\label{tab:ablation}\n"
            "  \\resizebox{\\textwidth}{!}{%\n"
            "  \\begin{tabular}{lccc}\n    \\toprule\n"
            "    \\textbf{MoE variant} & \\textbf{Avg.\\ accuracy} "
            "$\\uparrow$ & \\textbf{Avg.\\ forgetting} $\\downarrow$ "
            "& \\textbf{BWT} $\\uparrow$ \\\\\n    \\midrule")
        base = data["permuted_mnist"]["MoE"]
        variants = [("MoE ($\\alpha = 0.01$, main setting)", base)]
        for name in sorted(abl.keys()):
            lb = name.split("=")[1].rstrip(")")
            variants.append((f"MoE ($\\alpha = {lb}$)", abl[name]))
        for label, recs in variants:
            m_acc, s_acc = stats(recs, "avg_acc")
            m_for, s_for = stats(recs, "avg_forgetting")
            m_bwt, s_bwt = stats(recs, "bwt")
            abl_lines.append(f"    {label} & {fmt(m_acc, s_acc)} & "
                             f"{fmt(m_for, s_for)} & {fmt(m_bwt, s_bwt)} \\\\")
        abl_lines.append("    \\bottomrule\n  \\end{tabular}}%\n\\end{table}\n")
        out["07_table_ablation.tex"] = abl_lines

    for fname, content in out.items():
        with open(os.path.join(OUTDIR, fname), "w") as f:
            f.write("\n".join(content))
        print("wrote", fname)


if __name__ == "__main__":
    main()
