"""Render all figures for the invention disclosure from results/*.json
plus two illustrative traces (a fall episode and one day of zone heating)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Reference categorical palette (validated order) + text tokens.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlecolor": INK,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc="#eef4fc", ec=BLUE, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, text=None, color=INK2, off=(0, 0.12), style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.2))
    if text:
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, ha="center",
                va="center", fontsize=7, color=color, style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.grid(False)
    # Edge tier
    ax.add_patch(FancyBboxPatch((0.1, 0.2), 2.9, 5.2, boxstyle="round,pad=0.02",
                                fc="#fafaf8", ec=MUTED, lw=1, ls="--"))
    ax.text(1.55, 5.2, "EDGE TIER (battery sensor nodes)", ha="center", fontsize=8,
            weight="bold", color=INK2)
    for i, s in enumerate(["PIR / presence", "Power / water", "CO₂ / sound / light",
                           "Wearable IMU", "Bed load"]):
        y = 4.35 - i * 0.82
        box(ax, 0.3, y, 1.25, 0.6, s, fc="white", ec=MUTED, fs=7)
        box(ax, 1.65, y, 1.2, 0.6, "Twin predictor\n+ dead-band θᵢ(t)\n(gate 101)", fs=6.2)
    # Gateway tier
    ax.add_patch(FancyBboxPatch((3.5, 0.2), 3.4, 5.2, boxstyle="round,pad=0.02",
                                fc="#fafaf8", ec=MUTED, lw=1, ls="--"))
    ax.text(5.2, 5.2, "AMBIENT GATEWAY (edge AI)", ha="center", fontsize=8,
            weight="bold", color=INK2)
    box(ax, 3.7, 4.0, 3.0, 0.8, "Synchronised twin mirror 201\n(reconstructs x̂ for silent nodes)", fs=7)
    box(ax, 3.7, 2.9, 3.0, 0.8, "Silence-aware intent engine 202\n(interval likelihood + Bayesian filter)", fs=7)
    box(ax, 3.7, 1.8, 3.0, 0.8, "Attention-vector generator 203\nDᵢ(t) → θᵢ(t), safety floor", fs=7)
    box(ax, 3.7, 0.5, 3.0, 0.9, "Counterfactual digital-twin\nshield 301 (CVaR veto/repair)", fc="#fdf0ea", ec=ORANGE, fs=7)
    # Actuation tier
    ax.add_patch(FancyBboxPatch((7.4, 0.2), 2.5, 5.2, boxstyle="round,pad=0.02",
                                fc="#fafaf8", ec=MUTED, lw=1, ls="--"))
    ax.text(8.65, 5.2, "SERVICES / ACTUATION", ha="center", fontsize=8, weight="bold", color=INK2)
    box(ax, 7.6, 4.0, 2.1, 0.8, "Emergency alert\n(fall → caregiver)", fc="#fdecec", ec=RED, fs=7)
    box(ax, 7.6, 2.9, 2.1, 0.8, "Agentic planner\n(LLM / MPC / rules)", fc="white", ec=MUTED, fs=7)
    box(ax, 7.6, 0.5, 2.1, 0.9, "Zone HVAC, lights,\nappliances", fc="white", ec=MUTED, fs=7)
    # Arrows
    arrow(ax, 2.85, 3.5, 3.7, 4.3)
    ax.text(3.27, 3.35, "sparse\nuplink\n(|r|>θ)", ha="center", fontsize=6.5, color=BLUE, style="italic")
    arrow(ax, 3.7, 2.2, 2.85, 1.9)
    ax.text(3.25, 1.35, "attention\nbeacon\nθ(t)", ha="center", fontsize=6.5, color=ORANGE, style="italic")
    arrow(ax, 5.2, 4.0, 5.2, 3.7)
    arrow(ax, 5.2, 2.9, 5.2, 2.6)
    arrow(ax, 6.7, 3.4, 7.6, 4.3)
    ax.text(7.0, 4.15, "P(FALL)>τ", ha="center", fontsize=6.5, color=RED, style="italic", rotation=45)
    arrow(ax, 6.7, 3.2, 7.6, 3.3)
    ax.text(7.15, 3.05, "p(t)", ha="center", fontsize=6.5, color=INK2, style="italic")
    arrow(ax, 7.6, 3.0, 6.7, 1.2)
    ax.text(7.45, 2.1, "proposed\nplan", ha="center", fontsize=6.5, color=INK2, style="italic")
    arrow(ax, 6.7, 0.8, 7.6, 0.8)
    ax.text(7.15, 0.3, "certified /\nrepaired action", ha="center", fontsize=6.5, color=ORANGE, style="italic")
    save(fig, "fig1_architecture.png")


def flowchart():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.grid(False)
    W, H = 2.3, 0.75
    X = [0.1, 2.6, 5.1, 7.6]
    steps = [
        ("S1  Sample xᵢ(t)", X[0], 3.7), ("S2  Twin prediction x̂ᵢ(t)\n(identical at gateway)", X[1], 3.7),
        ("S3  |xᵢ − x̂ᵢ| > θᵢ(t) ?", X[2], 3.7),
        ("S4a  Transmit xᵢ(t)\nre-sync twin", X[3], 4.45), ("S4b  Stay silent\n(radio off)", X[3], 3.25),
        ("S5  Likelihood: N(xᵢ) if sent;\n∫ over [x̂ᵢ ± θᵢ] if silent", X[3], 2.05),
        ("S6  Bayesian intent\nupdate → p(t)", X[3], 0.75),
        ("S7  Look-ahead q = p·A⁽ᴸ⁾\n+ safety floor on FALL", X[2], 0.75),
        ("S8  Dᵢ = Σₖ qₖ((μₖᵢ−μ̄ᵢ)/σₖᵢ)²\nθᵢ = θmin+(θmax−θmin)e^(−κDᵢ)", X[1], 0.75),
        ("S9  Quantise & beacon θ(t)\n(tighten now, relax lazily)", X[0], 0.75),
    ]
    for t, x, y in steps:
        box(ax, x, y, W, H, t, fs=7)
    m = H / 2
    arrow(ax, X[0] + W, 3.7 + m, X[1], 3.7 + m)
    arrow(ax, X[1] + W, 3.7 + m, X[2], 3.7 + m)
    arrow(ax, X[2] + W, 3.7 + m + 0.1, X[3], 4.45 + m, "yes", off=(-0.05, 0.15))
    arrow(ax, X[2] + W, 3.7 + m - 0.1, X[3], 3.25 + m, "no", off=(-0.05, -0.15))
    arrow(ax, X[3] + W / 2, 3.25, X[3] + W / 2, 2.05 + H)
    arrow(ax, X[3] + W / 2, 2.05, X[3] + W / 2, 0.75 + H)
    arrow(ax, X[3], 0.75 + m, X[2] + W, 0.75 + m)
    arrow(ax, X[2], 0.75 + m, X[1] + W, 0.75 + m)
    arrow(ax, X[1], 0.75 + m, X[0] + W, 0.75 + m)
    arrow(ax, X[0] + W / 2, 0.75 + H, X[0] + W / 2, 3.7, "next step t+1\n(θ(t) to all nodes)",
          color=ORANGE, off=(0.0, 0.0))
    ax.text(5, 0.25, "Edge-node loop (S1–S4) runs on every sensor; the gateway loop (S5–S9) closes the "
            "intent-uncertainty feedback that sets every node's dead-band.", ha="center",
            fontsize=7.5, color=INK2)
    save(fig, "fig2_method_flow.png")


# ---------------------------------------------------------------- results
def load():
    with open(os.path.join(ROOT, "results", "results.json")) as f:
        return json.load(f)


COL = {"Periodic": YELLOW, "Send-on-Delta": ORANGE, "Dual-Prediction": AQUA, "AURA": BLUE}


def pareto(res):
    g = res["gating"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for fam, c in COL.items():
        pts = sorted([(v["tx_rate"][0] * 100, v["macro_f1"][0]) for k, v in g.items()
                      if k.startswith(fam + "|")])
        x, y = zip(*pts)
        ax.plot(x, y, "-o", color=c, lw=2, ms=6, mec="white", mew=1.5,
                label=fam + (" (proposed)" if fam == "AURA" else ""), zorder=3 if fam == "AURA" else 2)
    o = g["Oracle (stream all)"]["macro_f1"][0]
    ax.axhline(o, color=MUTED, ls="--", lw=1)
    ax.text(0.9, o + 0.004, f"Oracle, 100 % traffic (F1 = {o:.3f})", color=INK2, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Uplink traffic (% of samples transmitted, log scale)")
    ax.set_ylabel("Intent recognition macro-F1")
    ax.set_title("Accuracy–traffic Pareto front (420 unseen home-days)", loc="left")
    ax.legend(loc="lower right")
    save(fig, "fig3_pareto.png")


def ablation(res):
    g = res["gating"]
    names = ["AURA (full)", "– interval (silence) likelihood", "– intent-coupled dead-bands (fixed 2σ)",
             "– look-ahead prediction", "– safety floor"]
    f1 = [g[n]["macro_f1"][0] for n in names]
    f1e = [g[n]["macro_f1"][1] for n in names]
    tx = [g[n]["tx_rate"][0] * 100 for n in names]
    rc = [g[n]["recall"][0] * 100 for n in names]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, v, lab, e in zip(axs, (f1, tx, rc), ("Macro-F1", "Uplink traffic (%)", "Fall recall (%)"),
                             (f1e, None, None)):
        ax.barh(y, v, color=[BLUE] + [MUTED] * 4, height=0.6, xerr=e,
                error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for yi, vi in zip(y, v):
            ax.text(vi, yi, f" {vi:.3f}" if lab == "Macro-F1" else f" {vi:.1f}", va="center",
                    fontsize=7, color=INK)
        ax.set_title(lab, loc="left")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(names)
    axs[0].set_xlim(min(f1) - 0.08, 1.0)
    save(fig, "fig5_ablation.png")


def lifetime(res):
    g = res["gating"]
    names = ["Oracle (stream all)", "Periodic (2 min)", "Periodic (4 min)", "Send-on-Delta (2σ)",
             "Dual-Prediction (2σ)", "AURA (proposed)"]
    m = [g[n]["life_mean"][0] / 365 for n in names]
    w = [g[n]["life_min"][0] / 365 for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(names))
    ax.bar(x - 0.19, m, 0.36, color=[MUTED] * 5 + [BLUE], label="mean node")
    ax.bar(x + 0.19, w, 0.36, color=[GRID] * 5 + ["#9cc2ef"], label="worst node",
           edgecolor=[MUTED] * 5 + [BLUE], lw=0.8)
    for xi, a, b in zip(x, m, w):
        ax.text(xi - 0.19, a, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + 0.19, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
    ax.set_ylabel("Projected CR2032 lifetime (years)")
    ax.set_title("Edge-node battery lifetime", loc="left")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_lifetime.png")


def energy_comfort(res):
    t = res["twin"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pts = {"Static schedule": (INK2, "s"), "Reactive occupancy (ECO idle)": (ORANGE, "D"),
           "Reactive occupancy (standby idle)": (YELLOW, "D"),
           "Greedy agent (most-likely intent)": (VIOLET, "^"),
           "AURA shield – risk-neutral (mean)": (AQUA, "o")}
    for n, (c, mk) in pts.items():
        ax.plot(t[n]["energy_kwh_day"][0], t[n]["discomfort_kh_day"][0], mk, color=c, ms=9,
                mec="white", mew=1.5, label=n)
    bud = sorted([(float(k.split("|")[1]), v) for k, v in t.items() if k.startswith("budget|")]
                 + [(0.005, t["AURA shield (CVaR, proposed)"])])
    ax.plot([v["energy_kwh_day"][0] for _, v in bud], [v["discomfort_kh_day"][0] for _, v in bud],
            "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1.5, label="AURA shield (CVaR), budget sweep")
    for b, v in bud:
        ax.annotate(f"β={b}", (v["energy_kwh_day"][0], v["discomfort_kh_day"][0]),
                    xytext=(5, 4), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Heating energy (kWh/day)")
    ax.set_ylabel("Occupant discomfort (K·h/day, log)")
    ax.set_title("Energy–comfort trade-off, 4-zone dwelling (lower-left is better)", loc="left")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "fig7_energy_comfort.png")


def per_class(res):
    from aura.config import ACTIVITIES
    g = res["gating"]
    names = [("Send-on-Delta (2σ)", ORANGE), ("Dual-Prediction (2σ)", AQUA),
             ("AURA (proposed)", BLUE), ("Oracle (stream all)", MUTED)]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    x = np.arange(len(ACTIVITIES))
    w = 0.2
    for i, (n, c) in enumerate(names):
        ax.bar(x + (i - 1.5) * w, g[n]["per_class_f1"], w * 0.9, color=c, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIVITIES)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-intent F1", loc="left")
    ax.legend(ncol=4, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.08))
    ax.grid(axis="x", visible=False)
    save(fig, "fig8_per_class.png")


# ------------------------------------------------------- illustrative traces
def traces():
    from aura.simulator import simulate_home
    from aura.intent import IntentHMM
    from aura.gating import run_policy
    from aura.twin import run_controller, BATH, ZONE_OF
    from aura.config import FALL, STEP_S, HYGIENE
    train = [simulate_home(s, 14) for s in range(12)]
    hmm = IntentHMM().fit(train)
    home = next(h for h in (simulate_home(2000 + i, 4) for i in range(40))
                if len(h.fall_onsets) and h.fall_onsets[0] > 200)
    ra = run_policy(home, hmm, dict(kind="aura"), keep_theta=True)
    rs = run_policy(home, hmm, dict(kind="sod", c=2.0))
    ro = run_policy(home, hmm, dict(kind="oracle"))
    s0 = int(home.fall_onsets[0])
    w = slice(s0 - 40, s0 + 60)
    tm = (np.arange(w.start, w.stop) - s0) * STEP_S / 60
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    axs[0].plot(tm, home.x[w, 0], color=BLUE, lw=1.5, label="PIR living (events/min)")
    axs[0].plot(tm, home.x[w, 9] * 3, color=ORANGE, lw=1.5, label="wearable |a| ×3 (g)")
    axs[0].set_title("(a) Raw sensor streams around a fall (t = 0)", loc="left")
    axs[0].legend(fontsize=7, loc="upper right")
    for i, (si, c) in enumerate([(0, BLUE), (9, ORANGE), (6, AQUA), (10, VIOLET)]):
        axs[1].step(tm, ra["theta_trace"][w, si], color=c, lw=1.8, where="post",
                    label=["PIR living", "wearable", "sound", "bed load"][i])
    axs[1].axhline(2.0, color=MUTED, ls="--", lw=1)
    axs[1].text(tm[0], 2.08, "fixed send-on-delta dead-band (2σ)", fontsize=6.5, color=INK2)
    axs[1].set_ylabel("θᵢ / σᵢ")
    axs[1].set_title("(b) AURA attention vector: dead-bands tighten when intent is ambiguous", loc="left")
    axs[1].legend(fontsize=7, ncol=4, loc="upper right")
    for r, c, lab in [(ro, MUTED, "Oracle (100 % traffic)"), (rs, ORANGE, "Send-on-Delta"),
                      (ra, BLUE, "AURA")]:
        axs[2].plot(tm, r["posts"][w, FALL], color=c, lw=2 if c == BLUE else 1.5, label=lab,
                    ls="--" if c == MUTED else "-")
    axs[2].axhline(0.5, color=RED, lw=0.8, ls=":")
    axs[2].text(tm[0], 0.53, "alarm threshold", fontsize=6.5, color=RED)
    axs[2].set_ylabel("P(FALL)")
    axs[2].set_xlabel("minutes from fall onset")
    axs[2].set_title("(c) Gateway emergency posterior", loc="left")
    axs[2].legend(fontsize=7, loc="center right")
    save(fig, "fig4_fall_trace.png")

    # One day of bathroom temperature
    home = simulate_home(2001, 3)
    posts = run_policy(home, hmm, dict(kind="aura"))["posts"]
    d0, d1 = 2880, 2 * 2880
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    hrs = np.arange(d1 - d0) * STEP_S / 3600
    for ctrl, c, lab in [("schedule", MUTED, "Static schedule"), ("reactive", ORANGE, "Reactive occupancy"),
                         ("shield", BLUE, "AURA shield")]:
        o = run_controller(home, posts, hmm, ctrl, seed=7)
        ax.plot(hrs, o["temps"][d0:d1, BATH], color=c, lw=1.8 if c == BLUE else 1.3, label=lab)
    use = home.labels[d0:d1] == HYGIENE
    ymin, ymax = ax.get_ylim()
    ax.fill_between(hrs, ymin, ymax, where=use, color=YELLOW, alpha=0.25, lw=0,
                    label="bathroom in use")
    ax.axhline(20, color=RED, lw=0.8, ls=":")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, 24)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("Bathroom temperature (°C)")
    ax.set_title("Pre-conditioning of the bathroom zone over one day", loc="left")
    ax.legend(fontsize=7, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.08))
    save(fig, "fig9_bathroom_day.png")


if __name__ == "__main__":
    architecture()
    flowchart()
    if os.path.exists(os.path.join(ROOT, "results", "results.json")):
        res = load()
        pareto(res)
        ablation(res)
        lifetime(res)
        energy_comfort(res)
        per_class(res)
    if "--no-traces" not in sys.argv:
        traces()
