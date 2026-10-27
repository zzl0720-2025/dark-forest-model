"""Run the 11 x 20 experiment, retain event records, and build figures/replays.

From the project directory: .venv/bin/python code/run_all.py
"""
from __future__ import annotations

import csv
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import Event, parameters_from_config, simulate, equilibrium, ode_solution, stationary_distribution, rhs

BLUE = "#245c88"
TEAL = "#00877d"
ORANGE = "#c46d24"
INK = "#243344"
GRAY = "#697987"
T19 = 2.0930240544


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.titleweight": "medium", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#a7b2bd",
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": GRAY, "ytick.color": GRAY,
                         "grid.color": "#dce3e8", "grid.linewidth": .6,
                         "legend.frameon": False, "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, name):
    destination = ROOT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(destination / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plots(config, results, summaries):
    style()
    probs = config["discovery_probabilities"]
    selected = config["replay_probabilities"]
    base = parameters_from_config(config, .5)
    times = np.asarray(results[0.0][0].sample_times)
    ps = np.asarray(probs)
    stars = np.asarray([base.capacity*equilibrium(replace(base, discovery=p)) for p in probs])
    finite = np.asarray([s["stationary_mean"] for s in summaries])
    means = np.asarray([s["mean_final"] for s in summaries])
    errors = np.asarray([s["ci95_half_width"] for s in summaries])

    fig, ax = plt.subplots(figsize=(7.4, 4.15), layout="constrained")
    ax.plot(ps, stars, color=ORANGE, lw=2, label="Mean-field equilibrium")
    ax.plot(ps, finite, color=TEAL, lw=1.8, ls="--", label="Exact finite-population stationary mean")
    ax.errorbar(ps, means, yerr=errors, fmt="o", color=BLUE, ms=5, capsize=3,
                lw=1.2, label="ABM at T = 500: mean and 95% CI (20 runs)")
    ax.set(xlabel="Discovery probability per occupied target, p", ylabel="Number of civilizations",
           title="A more detectable galaxy supports fewer civilizations", xlim=(-.025, 1.025))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "01_discovery_sweep")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.25), sharex=True, sharey=True, layout="constrained")
    for ax, p in zip(axes, selected):
        trials = results[p]
        all_counts = np.asarray([trial.sample_counts for trial in trials])
        for sample in all_counts[:4]:
            ax.step(times, sample, where="post", color=BLUE, alpha=.15, lw=.65)
        average = all_counts.mean(axis=0)
        half_width = T19*all_counts.std(axis=0, ddof=1)/math.sqrt(len(trials))
        ax.fill_between(times, np.maximum(average-half_width,0), average+half_width,
                        color=BLUE, alpha=.15, linewidth=0)
        ax.plot(times, average, color=BLUE, lw=1.6, label="ABM mean (20 runs)")
        par = replace(base, discovery=p)
        ode = [par.capacity*ode_solution(t,par.initial/par.capacity,par) for t in times]
        ax.plot(times, ode, color=ORANGE, lw=1.6, label="Mean-field ODE")
        ax.axhline(summaries[probs.index(p)]["stationary_mean"], color=TEAL, ls="--", lw=1,
                   label="Finite stationary mean")
        ax.set(title=f"p = {p:g}", xlabel="Time (model units)", xlim=(0,config["duration"]), ylim=(0,27))
        ax.grid(axis="y")
    axes[0].set_ylabel("Number of civilizations")
    axes[-1].legend(fontsize=7, loc="upper right")
    save(fig, "02_time_series")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), layout="constrained")
    for p,color in zip(selected,(BLUE,TEAL,ORANGE)):
        par=replace(base,discovery=p)
        grid=np.linspace(0,.15,300)
        axes[0].plot(grid,[rhs(x,par) for x in grid],color=color,label=f"p = {p:g}")
        root=equilibrium(par)
        axes[0].scatter([root],[0],color=color,s=30,zorder=5)
        for direction,xx in ((1,root*.45),(-1,(root+.15)/2)):
            axes[0].annotate("", xy=(xx+direction*.009,0), xytext=(xx-direction*.009,0),
                             arrowprops={"arrowstyle":"->","color":color,"lw":1.2})
    axes[0].axhline(0,color=GRAY,lw=.7,zorder=0)
    axes[0].set(xlabel="Occupied fraction, x",ylabel="Rate of change, dx/dt",title="One stable equilibrium at each p")
    axes[0].legend(fontsize=8)
    for initial,color in zip((0,.2,.8),(TEAL,BLUE,ORANGE)):
        axes[1].plot(times,[ode_solution(t,initial,base) for t in times],color=color,label=f"x(0) = {initial:g}")
    axes[1].axhline(equilibrium(base),color=GRAY,lw=.8,ls=":")
    axes[1].set(xlabel="Time (model units)",ylabel="Occupied fraction, x",title="Initial conditions change the transient (p = 0.5)")
    axes[1].legend(fontsize=8)
    save(fig,"03_phase_and_initial_conditions")

    fig,axes=plt.subplots(1,3,figsize=(10,3.15),sharex=True,layout="constrained")
    grid=np.arange(0,19)
    for ax,p in zip(axes,selected):
        finals=[trial.final_count for trial in results[p]]
        frequencies=[finals.count(int(n))/len(finals) for n in grid]
        law=stationary_distribution(replace(base,discovery=p))
        ax.bar(grid,frequencies,color=BLUE,alpha=.5,width=.8,label="20 final counts")
        ax.plot(grid,law[:len(grid)],color=TEAL,marker=".",lw=1.3,label="Exact stationary law")
        ax.set(title=f"p = {p:g}",xlabel="Civilizations at T = 500",xlim=(-.6,18.6))
        ax.set_xticks([0,4,8,12,16])
        ax.grid(axis="y")
    axes[0].set_ylabel("Relative frequency / probability")
    axes[-1].legend(fontsize=7)
    save(fig,"04_final_count_distributions")

    fig,axes=plt.subplots(1,2,figsize=(8.4,3.4),layout="constrained")
    sizes=[25,100,400,1600]
    finite_fractions=[]
    for m in sizes:
        par=replace(base,side=math.isqrt(m),initial=min(base.initial,m))
        law=stationary_distribution(par)
        finite_fractions.append(sum(n*w for n,w in enumerate(law))/m)
    axes[0].plot(sizes,finite_fractions,"o-",color=TEAL,label="Exact stationary occupied fraction")
    axes[0].axhline(equilibrium(base),color=ORANGE,ls="--",label="Mean-field equilibrium")
    axes[0].set(xscale="log",xlabel="Number of stars, M (log scale)",ylabel="Occupied fraction",title="Finite-size correction shrinks with galaxy size")
    axes[0].set_xticks(sizes,labels=[str(m) for m in sizes])
    axes[0].legend(fontsize=7)
    for key,label,color in (("mean_births","Births",TEAL),("mean_attacks","Attack deaths",ORANGE),("mean_collapses","Collapse deaths",BLUE)):
        axes[1].plot(ps,[row[key] for row in summaries],"o-",ms=3,color=color,label=label)
    axes[1].set(xlabel="Discovery probability, p",ylabel="Mean events per run (0 to T)",title="Attack deaths replace part of natural turnover")
    axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(axis="y")
    save(fig,"05_finite_size_and_turnover")


def replay_payload(config, results):
    kind_code={"birth":0,"collapse":1,"attack":2,"empty_search":3,"undetected_search":4}
    runs=[]
    for p in config["replay_probabilities"]:
        trial=results[p][0]  # The first repetition, selected before seeing outcomes.
        runs.append({"p":p,"seed":trial.seed,"initial":trial.initial_sites,
                     "events":[[round(e.time,7),kind_code[e.kind],e.source,e.target,e.count_after] for e in trial.events],
                     "ode":[round(trial.parameters.capacity*ode_solution(t,trial.parameters.initial/trial.parameters.capacity,trial.parameters),6)
                            for t in trial.sample_times],
                     "times":trial.sample_times,"counts":trial.sample_counts})
    return {"side":config["side"],"duration":config["duration"],"runs":runs}


def main():
    config_path=ROOT/"config"/"experiment.json"
    config=json.loads(config_path.read_text())
    data=ROOT/"data"
    events_dir=data/"events"
    events_dir.mkdir(parents=True,exist_ok=True)
    (ROOT/"visualization").mkdir(exist_ok=True)
    results={}
    run_rows=[]
    summaries=[]
    time_rows=[]
    stationary_rows=[]
    for index,p in enumerate(config["discovery_probabilities"]):
        par=parameters_from_config(config,p)
        trials=[]
        for repetition in range(1,config["repetitions"]+1):
            seed=config["base_seed"]+10000*index+repetition
            trial=simulate(par,seed)
            trials.append(trial)
            filename=f"p{index:02d}_run{repetition:02d}.csv.gz"
            with gzip.open(events_dir/filename,"wt",newline="",encoding="utf-8") as stream:
                writer=csv.writer(stream)
                writer.writerow([field.name for field in fields(Event)])
                writer.writerows(tuple(asdict(event).values()) for event in trial.events)
            run_rows.append({"p":p,"repetition":repetition,"seed":seed,
                             "initial_count":par.initial,"final_count":trial.final_count,
                             "births":trial.births,"collapses":trial.collapses,"attacks":trial.attacks,
                             "searches":trial.searches,"empty_time":trial.empty_time,
                             "empty_at_T":int(trial.final_count==0),
                             "ever_empty":int(trial.empty_time>0),"event_file":"events/"+filename})
        results[p]=trials
        outcomes=[trial.final_count for trial in trials]
        mean=statistics.mean(outcomes)
        sd=statistics.stdev(outcomes)
        se=sd/math.sqrt(len(outcomes))
        law=stationary_distribution(par)
        finite_mean=sum(n*value for n,value in enumerate(law))
        summaries.append({"p":p,"repetitions":len(trials),"mean_final":mean,"sd_final":sd,
                          "se_final":se,"ci95_half_width":T19*se,"min_final":min(outcomes),"max_final":max(outcomes),
                          "empty_at_T_fraction":sum(n==0 for n in outcomes)/len(outcomes),
                          "ever_empty_fraction":statistics.mean(trial.empty_time>0 for trial in trials),
                          "mean_empty_time":statistics.mean(trial.empty_time for trial in trials),
                          "mean_births":statistics.mean(trial.births for trial in trials),
                          "mean_collapses":statistics.mean(trial.collapses for trial in trials),
                          "mean_attacks":statistics.mean(trial.attacks for trial in trials),
                          "ode_at_T":par.capacity*ode_solution(par.duration,par.initial/par.capacity,par),
                          "ode_equilibrium":par.capacity*equilibrium(par),"stationary_mean":finite_mean,
                          "stationary_empty_probability":law[0]})
        for sample_index,t in enumerate(trials[0].sample_times):
            for repetition,trial in enumerate(trials,1):
                time_rows.append({"p":p,"repetition":repetition,"seed":trial.seed,"time":t,"count":trial.sample_counts[sample_index]})
        stationary_rows.extend({"p":p,"count":n,"probability":value} for n,value in enumerate(law))
        print(f"p={p:.1f}: mean final={mean:.2f}; SD={sd:.2f}; finite stationary mean={finite_mean:.3f}",flush=True)
    write_csv(data/"runs.csv",run_rows)
    write_csv(data/"summary.csv",summaries)
    write_csv(data/"trajectories.csv",time_rows)
    write_csv(data/"stationary_distributions.csv",stationary_rows)
    payload=replay_payload(config,results)
    (data/"replay.json").write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")
    template=(ROOT/"code"/"viewer_template.html").read_text(encoding="utf-8")
    viewer=template.replace("__REPLAY_DATA__",json.dumps(payload,separators=(",",":")))
    (ROOT/"visualization"/"index.html").write_text(viewer,encoding="utf-8")
    plots(config,results,summaries)
    versions={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"matplotlib":matplotlib.__version__}
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"config":config,"versions":versions,
              "seed_rule":"base_seed + 10000 * probability_index + repetition (1..20)",
              "sampling":"Counts are right-continuous, sampled every 0.5 model time units.",
              "ci":"pointwise two-sided Student t interval, df=19; not a simultaneous band",
              "replay_selection":"first repetition at p=0, 0.5, 1; fixed before inspecting outcomes",
              "files":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((ROOT/"code").glob("*")) if path.is_file()}}
    (data/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    # LaTeX table generated directly from the same experiment summary.
    lines=[r"\begin{tabular}{rrrrrr}",r"\toprule",r"$p$ & Mean $N(T)$ & SD & 95\% half-width & ODE $N^*$ & Finite $E_\pi[N]$ \\",r"\midrule"]
    for row in summaries:
        lines.append(f"{row['p']:.1f} & {row['mean_final']:.2f} & {row['sd_final']:.2f} & {row['ci95_half_width']:.2f} & {row['ode_equilibrium']:.2f} & {row['stationary_mean']:.2f} \\\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    # In the author's working directory only, refresh the report's assets.
    # A standalone code distribution needs no report sources or TeX installation.
    if (ROOT/"report"/"main.tex").exists():
        generated=ROOT/"report"/"generated"
        generated.mkdir(parents=True,exist_ok=True)
        (generated/"results_table.tex").write_text("\n".join(lines)+"\n")
        report_figures=ROOT/"report"/"figures"
        report_figures.mkdir(exist_ok=True)
        for figure in (ROOT/"figures").glob("*"):
            if figure.is_file():
                shutil.copy2(figure,report_figures/figure.name)
    print("Saved 220 trials, all event logs, summaries, figures, and an offline replay.",flush=True)


if __name__=="__main__":
    main()
