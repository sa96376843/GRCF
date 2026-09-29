#!/usr/bin/env python
"""Generate the verified result tables used by manuscript.tex.

Only complete experiment files are consumed: results/main.json and
results/ablation.json.  Incomplete robustness runs are intentionally excluded.
"""
from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PAPER = BASE.parent
TAB = PAPER / "tables"
TAB.mkdir(exist_ok=True)


def load_json(name: str):
    return json.loads((BASE / "results" / f"{name}.json").read_text(encoding="utf-8"))


def aggregate(main, dataset, method, metric):
    values = [main[dataset][seed][method][metric] for seed in sorted(main[dataset])]
    return statistics.mean(values), statistics.pstdev(values)


def write_data_table():
    xtx_path = BASE / "data" / "raw" / "xuetangx" / "data.csv"
    xtx_users, xtx_courses, xtx_dates = set(), set(), []
    xtx_count = 0
    with xtx_path.open("r", encoding="gbk", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            xtx_count += 1
            xtx_users.add(row[0])
            xtx_courses.add(row[2])
            xtx_dates.append(datetime.strptime(row[1], "%Y/%m/%d %H:%M"))
    mars_path = BASE / "data" / "raw" / "mars_explicit_fr.tab"
    mars_users, mars_courses = set(), set()
    mars_count = mars_positive = 0
    with mars_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        user_col = next((x for x in fields if "user" in x.lower()), fields[0])
        item_col = next((x for x in fields if any(t in x.lower() for t in ["item", "resource", "course"])), fields[1])
        rating_col = next((x for x in fields if "rating" in x.lower() or "score" in x.lower()), fields[-1])
        for row in reader:
            mars_count += 1
            mars_users.add(row[user_col])
            mars_courses.add(row[item_col])
            try:
                mars_positive += float(row[rating_col]) >= 6
            except ValueError:
                pass
    days = (max(xtx_dates) - min(xtx_dates)).days
    positive = 100.0 * mars_positive / mars_count
    text = rf"""\begin{{table*}}[t]
\caption{{Statistics of the two public online-learning datasets.}}
\label{{tab:data}}
\centering
\begin{{tabular}}{{lcc}}
\toprule
 & \textbf{{XTX (XuetangX)}} & \textbf{{MARS}} \\
\midrule
Interactions & {xtx_count:,} enrolments & {mars_count:,} ratings \\
Users & {len(xtx_users):,} & {len(mars_users):,} \\
Items & {len(xtx_courses):,} courses & {len(mars_courses):,} resources \\
Feedback & implicit enrolment & explicit 0--10 rating \\
Positive definition & enrolled & rating $\geq 6$ ({positive:.1f}\%) \\
Time information & {days} days & not provided \\
Text fields & course name and category & name and description \\
Training negatives & two sampled per positive & observed ratings below 6 \\
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (TAB / "tab_data.tex").write_text(text, encoding="utf-8")


def write_main_table():
    main = load_json("main")
    methods = ["LR", "RF", "MLP", "WideDeep", "NCF", "LightGCN", "SGL", "GRA-MLOF", "GRCF"]
    best_auc = {d: max(aggregate(main, d, m, "auc")[0] for m in methods) for d in ["XTX", "MARS"]}
    best_f1 = {d: max(aggregate(main, d, m, "f1")[0] for m in methods) for d in ["XTX", "MARS"]}
    rows = []
    for method in methods:
        cells = [method]
        for dataset in ["XTX", "MARS"]:
            auc, auc_sd = aggregate(main, dataset, method, "auc")
            f1, f1_sd = aggregate(main, dataset, method, "f1")
            auc_s = f"{auc:.4f}$\\pm${auc_sd:.4f}"
            f1_s = f"{f1:.4f}$\\pm${f1_sd:.4f}"
            if abs(auc - best_auc[dataset]) < 1e-12:
                auc_s = rf"\textbf{{{auc_s}}}"
            if abs(f1 - best_f1[dataset]) < 1e-12:
                f1_s = rf"\textbf{{{f1_s}}}"
            cells.extend([auc_s, f1_s])
        rows.append(" & ".join(cells) + r" \\")
    text = r"""\begin{table*}[t]
\caption{Main recommendation results (mean $\pm$ standard deviation over three seeds using the same random 8:1:1 split). Bold marks the highest mean in each dataset and metric.}
\label{tab:main}
\centering
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{XTX}} & \multicolumn{2}{c}{\textbf{MARS}}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Method} & \textbf{AUC} & \textbf{F1} & \textbf{AUC} & \textbf{F1}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (TAB / "tab_main.tex").write_text(text, encoding="utf-8")


def write_ablation_table():
    ab = load_json("ablation")
    variants = [
        ("full", "GRCF (full)"),
        ("no_cl", "without contrastive loss"),
        ("no_gate", "without gated fusion"),
        ("no_evd", "without evidential head"),
        ("no_v1", "without V1 interactions"),
        ("no_v2", "without V2 attributes"),
        ("no_v3", "without V3 text"),
        ("no_ewgra", "without EW-GRA"),
    ]
    values = {}
    for key, label in variants:
        aucs = [v["auc"] for k, v in sorted(ab.items()) if k.startswith(key + "|")]
        values[key] = (statistics.mean(aucs), statistics.pstdev(aucs), label)
    full = values["full"][0]
    rows = []
    for key, _ in variants:
        mean, sd, label = values[key]
        delta = "--" if key == "full" else f"{mean - full:+.4f}"
        rows.append(f"{label} & {mean:.4f}$\\pm${sd:.4f} & {delta}" + r" \\")
    text = r"""\begin{table*}[t]
\caption{Ablation results on XTX (AUC, mean $\pm$ standard deviation over two seeds). $\Delta$ is relative to the full configuration.}
\label{tab:ablation}
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{AUC} & $\boldsymbol{\Delta}$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (TAB / "tab_ablation.tex").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    write_data_table()
    write_main_table()
    write_ablation_table()
    print("integrated tables generated")
