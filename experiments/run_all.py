"""Full experimental campaign for the AURA invention disclosure.

Protocol
--------
* Gateway intent model trained on 12 homes x 14 days (seeds 0-11).
* AURA hyper-parameters were tuned on 3 separate validation homes
  (seeds 1000-1002) that are NOT used below.
* Test: 5 independent repetitions x 6 unseen homes x 14 days
  = 30 home-runs, 420 home-days, 1.21 M sensing steps per policy.

Outputs: results/results.json, results/*.csv and figures/*.png
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aura.simulator import simulate_home                    # noqa: E402
from aura.intent import IntentHMM                           # noqa: E402
from aura.gating import run_policy                          # noqa: E402
from aura.metrics import recognition, fall_detection        # noqa: E402
from aura.twin import run_controller                        # noqa: E402
from aura.config import STEPS_PER_DAY, S                    # noqa: E402

DAYS = 14
TRAIN = [simulate_home(s, DAYS) for s in range(12)]
HMM = IntentHMM().fit(TRAIN)
TEST_SEEDS = [2000 + 100 * r + i for r in range(5) for i in range(6)]
TWIN_SEEDS = TEST_SEEDS[:18]

MAIN = {
    "Oracle (stream all)": dict(kind="oracle"),
    "Periodic (2 min)": dict(kind="periodic", period=4),
    "Periodic (4 min)": dict(kind="periodic", period=8),
    "Send-on-Delta (2σ)": dict(kind="sod", c=2.0),
    "Dual-Prediction (2σ)": dict(kind="dps", c=2.0),
    "AURA (proposed)": dict(kind="aura"),
}
ABLATION = {
    "AURA (full)": dict(kind="aura"),
    "– interval (silence) likelihood": dict(kind="aura", interval=False),
    "– intent-coupled dead-bands (fixed 2σ)": dict(kind="aura", adaptive=False, c_fixed=2.0),
    "– look-ahead prediction": dict(kind="aura", lookahead=False),
    "– safety floor": dict(kind="aura", safety_floor=0.0),
}
SWEEP = {}
for p in (2, 4, 8, 16, 32):
    SWEEP[f"Periodic|{p}"] = dict(kind="periodic", period=p)
for c in (1.0, 1.5, 2.0, 2.5, 3.0):
    SWEEP[f"Send-on-Delta|{c}"] = dict(kind="sod", c=c)
for c in (1.0, 1.5, 2.0, 2.5, 3.0):
    SWEEP[f"Dual-Prediction|{c}"] = dict(kind="dps", c=c)
for cm in (2.0, 3.0, 4.0, 6.0, 8.0):
    SWEEP[f"AURA|{cm}"] = dict(kind="aura", c_max=cm)


def eval_gating(args):
    name, pol, seed = args
    home = simulate_home(seed, DAYS)
    t0 = time.perf_counter()
    r = run_policy(home, HMM, pol)
    dt = time.perf_counter() - t0
    rec = recognition(home.labels, r["posts"])
    fd = fall_detection(home.labels, r["posts"], home.fall_onsets)
    days = home.days
    return dict(name=name, seed=seed, **rec, recall=fd["recall"],
                delays=fd["delays"], n_falls=len(home.fall_onsets),
                fa_per_day=fd["fa_per_day"], tx_rate=r["tx_rate"],
                pkts_day=float((r["tx"].sum() + r["events"]) / days),
                beacons_day=r["beacons"] / days,
                life_mean=float(r["life_days"].mean()),
                life_min=float(r["life_days"].min()),
                us_per_step=1e6 * dt / r["steps"],
                per_class=per_class_f1(home.labels, r["posts"]))


def per_class_f1(y, posts):
    from sklearn.metrics import f1_score
    return f1_score(y, posts.argmax(1), average=None, labels=list(range(8)),
                    zero_division=0).tolist()


TWIN_JOBS = {
    "Static schedule": ("schedule", {}, "aura"),
    "Reactive occupancy (ECO idle)": ("reactive", {}, "aura"),
    "Reactive occupancy (standby idle)": ("reactive_standby", {}, "aura"),
    "Greedy agent (most-likely intent)": ("greedy", {}, "aura"),
    "AURA shield – risk-neutral (mean)": ("shield", {"risk": "mean"}, "aura"),
    "AURA shield (CVaR, proposed)": ("shield", {}, "aura"),
    "AURA shield on Send-on-Delta stream": ("shield", {}, "sod"),
    "AURA shield on Oracle stream": ("shield", {}, "oracle"),
    "AURA shield, 25 % twin mismatch": ("shield", {"mismatch": 0.25}, "aura"),
    "budget|0.002": ("shield", {"budget": 0.002}, "aura"),
    "budget|0.01": ("shield", {"budget": 0.01}, "aura"),
    "budget|0.02": ("shield", {"budget": 0.02}, "aura"),
    "budget|0.05": ("shield", {"budget": 0.05}, "aura"),
}
STREAMS = {"aura": dict(kind="aura"), "sod": dict(kind="sod", c=2.0),
           "oracle": dict(kind="oracle")}


def eval_twin(args):
    name, seed = args
    ctrl, kw, stream = TWIN_JOBS[name]
    home = simulate_home(seed, DAYS)
    posts = run_policy(home, HMM, STREAMS[stream])["posts"]
    out = run_controller(home, posts, HMM, ctrl, seed=seed, **kw)
    return dict(name=name, seed=seed, **{k: v for k, v in out.items()
                                          if not isinstance(v, np.ndarray)})


def summarise(rows, keys):
    out = {}
    names = list(dict.fromkeys(r["name"] for r in rows))
    for n in names:
        rr = [r for r in rows if r["name"] == n]
        d = {}
        for k in keys:
            v = np.array([r[k] for r in rr], dtype=float)
            d[k] = [float(np.nanmean(v)), float(np.nanstd(v))]
        if "delays" in rr[0]:
            dl = np.concatenate([r["delays"] for r in rr]) if any(r["delays"] for r in rr) else np.array([np.nan])
            nf = sum(r["n_falls"] for r in rr)
            d["delay_mean_s"] = float(np.mean(dl))
            d["delay_p95_s"] = float(np.percentile(dl, 95))
            d["falls_total"] = int(nf)
            d["falls_detected"] = int(round(sum(r["recall"] * r["n_falls"] for r in rr)))
            d["per_class_f1"] = np.mean([r["per_class"] for r in rr], 0).tolist()
        out[n] = d
    return out


def paired(rows, a, b, key):
    x = {r["seed"]: r[key] for r in rows if r["name"] == a}
    y = {r["seed"]: r[key] for r in rows if r["name"] == b}
    s = sorted(set(x) & set(y))
    xa, yb = np.array([x[i] for i in s]), np.array([y[i] for i in s])
    if np.allclose(xa, yb):
        return 1.0
    return float(wilcoxon(xa, yb).pvalue)


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    t0 = time.time()
    gkeys = ["macro_f1", "acc", "tx_rate", "pkts_day", "beacons_day", "life_mean",
             "life_min", "recall", "fa_per_day", "us_per_step"]
    jobs = [(n, p, s) for d in (MAIN, ABLATION, SWEEP) for n, p in d.items()
            for s in TEST_SEEDS]
    jobs = list({(j[0], j[2]): j for j in jobs}.values())
    with Pool(os.cpu_count()) as P:
        grows = P.map(eval_gating, jobs, chunksize=2)
        print(f"gating done {time.time() - t0:.0f}s", flush=True)
        trows = P.map(eval_twin, [(n, s) for n in TWIN_JOBS for s in
                                  (TWIN_SEEDS if not n.startswith("budget")
                                   else TWIN_SEEDS[:12])], chunksize=1)
        print(f"twin done {time.time() - t0:.0f}s", flush=True)

    res = dict(protocol=dict(train_homes=len(TRAIN), days=DAYS,
                             test_homes=len(TEST_SEEDS),
                             test_home_days=len(TEST_SEEDS) * DAYS,
                             steps_per_policy=len(TEST_SEEDS) * DAYS * STEPS_PER_DAY,
                             twin_homes=len(TWIN_SEEDS), sensors=S),
               gating=summarise(grows, gkeys))
    res["twin"] = summarise(trows, ["energy_kwh_day", "discomfort_kh_day",
                                    "pct_uncomfortable", "interventions_day",
                                    "fall_max_deficit", "veto_pct"])
    A = "AURA (proposed)"
    res["stats_gating"] = {b: dict(
        p_f1=paired(grows, A, b, "macro_f1"), p_tx=paired(grows, A, b, "tx_rate"),
        p_life=paired(grows, A, b, "life_mean"))
        for b in MAIN if b != A}
    Sh = "AURA shield (CVaR, proposed)"
    res["stats_twin"] = {b: dict(
        p_energy=paired(trows, Sh, b, "energy_kwh_day"),
        p_discomfort=paired(trows, Sh, b, "discomfort_kh_day"))
        for b in TWIN_JOBS if b != Sh and not b.startswith("budget")}
    with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(ROOT, "results", "gating_runs.json"), "w") as f:
        json.dump(grows, f)
    with open(os.path.join(ROOT, "results", "twin_runs.json"), "w") as f:
        json.dump(trows, f)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
