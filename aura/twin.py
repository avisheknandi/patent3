"""Counterfactual digital-twin shield for agentic ambient actuation (AURA C3).

Scenario: intent-driven multi-zone heating of an assisted-living dwelling
(bedroom, bathroom, kitchen, living room). Only the zone the occupant is (or
will be) using must be comfortable, so an ambient agent can save energy by
letting idle zones float - but every wrong or late guess leaves the occupant
in a cold room (and a fallen occupant on a cold floor).

The AURA shield rolls a thermal digital twin of every zone forward under each
candidate actuation plan *and* under many intent futures sampled from the
gateway posterior; plans whose conditional value-at-risk (CVaR) of discomfort
exceeds a budget are vetoed and the least-energy admissible plan is executed
(receding horizon). Baselines: static programmable schedule, reactive
occupancy control, and a greedy agent that plans on the single most-likely
intent trajectory.
"""
import numpy as np

from .config import (K, STEP_S, STEPS_PER_DAY, SLEEP, HYGIENE, COOK, EAT,
                     RELAX, WORK, FALL)

SLOT = 10                         # decision period in steps (5 min)
DT_SLOT = SLOT * STEP_S
ECO, NIGHT, COMFORT = 15.0, 18.0, 21.0
ZONES = ["bedroom", "bathroom", "kitchen", "living"]
Z = len(ZONES)
BED, BATH, KITCH, LIV = range(Z)
ZONE_OF = np.full(K, -1)
ZONE_OF[[SLEEP, HYGIENE, COOK, EAT, RELAX, WORK]] = [BED, BATH, KITCH, LIV, LIV, BED]
REQ_LEVEL = np.full(K, 20.0)      # minimum acceptable temperature in the used zone
REQ_LEVEL[SLEEP] = 17.0


def outdoor_temp(T, rng):
    h = (np.arange(T) % STEPS_PER_DAY) * STEP_S / 3600
    day = np.arange(T) // STEPS_PER_DAY
    daily = rng.normal(2.0, 2.5, size=day.max() + 1)[day]
    return daily + 4.0 * np.sin(2 * np.pi * (h - 9) / 24)


def true_zone(labels):
    """Zone occupied at each step; a fall happens in the zone used just before."""
    z = ZONE_OF[labels].copy()
    for t in np.flatnonzero(labels == FALL):
        z[t] = z[t - 1] if t else LIV
    return z


def required_matrix(states, fall_zone):
    """(n, H) intent futures -> (Z, n, H) minimum zone temperatures."""
    zone = np.where(states == FALL, fall_zone, ZONE_OF[states])
    lvl = REQ_LEVEL[states]
    return np.stack([np.where(zone == z, lvl, -np.inf) for z in range(Z)])


class Building:
    """Ground-truth multi-zone RC thermal plant with local P-thermostats."""

    def __init__(self, rng):
        self.UA = rng.uniform(35, 60, size=Z)          # W/K per zone
        self.C = rng.uniform(4e5, 9e5, size=Z)         # J/K per zone
        self.qmax = np.full(Z, 2500.0)                 # W per zone
        self.kp = 8000.0

    def step(self, T, Tout, Tset):
        q = np.clip(self.kp * (Tset - T), 0, self.qmax)
        T = T + STEP_S * ((Tout - T) * self.UA + q) / self.C
        return T, q


class Twin:
    """Slot-level analytic zone twin with deliberate parameter mismatch."""

    def __init__(self, bld, rng, mismatch=0.10):
        self.UA = bld.UA * (1 + rng.uniform(-mismatch, mismatch, size=Z))
        self.C = bld.C * (1 + rng.uniform(-mismatch, mismatch, size=Z))
        self.qmax = bld.qmax
        self.a = np.exp(-DT_SLOT * self.UA / self.C)

    def advance(self, z, T, Tout, Tset):
        a, UA, qm = self.a[z], self.UA[z], self.qmax[z]
        free = Tout + (T - Tout) * a
        full = Tout + qm / UA + (T - Tout - qm / UA) * a
        Tn = np.maximum(np.minimum(Tset, full), free)
        q = (Tn - free) * UA / (1 - a)
        return Tn, q * DT_SLOT / 3.6e6


STANDBY = 18.0


def candidate_plans(H, base_level):
    """Hold a setback level (STANDBY or ECO) for j slots, then the comfort
    level (pre-heat timing). Row 0 heats immediately; the last row is the
    all-ECO (cheapest) proposal."""
    js = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, H]
    rows = [np.full(H, base_level)]
    for setback in (STANDBY, ECO):
        if setback >= base_level:
            continue
        for j in js:
            p = np.full(H, base_level)
            p[:j] = setback
            rows.append(p)
    return np.array(rows)


def sample_futures(hmm, post, hour0, H, N, rng):
    """Sample N intent trajectories over H 5-min slots from the posterior."""
    states = np.empty((N, H), dtype=np.int64)
    s = rng.choice(K, size=N, p=post / post.sum())
    for j in range(H):
        h = int(hour0 + j * DT_SLOT / 3600) % 24
        cum = hmm.A_look[h][s].cumsum(1)
        s = (cum < rng.random((N, 1)) * cum[:, -1:]).sum(1)
        states[:, j] = s
    return states


def run_controller(home, posts, hmm, ctrl, seed, H=24, N=64, alpha=0.9,
                   budget=0.005, risk="cvar", mismatch=0.10):
    rng = np.random.default_rng(seed)
    bld = Building(rng)
    twin = Twin(bld, rng, mismatch)
    T_all = len(home.labels)
    Tout = outdoor_temp(T_all, rng)
    T = np.full(Z, COMFORT)
    Tset = np.full(Z, COMFORT)
    energy = 0.0
    temps = np.empty((T_all, Z))
    setp = np.empty((T_all, Z))
    last_zone = LIV
    vetoes = 0
    decisions = 0
    for t in range(T_all):
        k_hat = int(posts[t].argmax())
        if ZONE_OF[k_hat] >= 0:
            last_zone = ZONE_OF[k_hat]
        if t % SLOT == 0:
            hour = (t % STEPS_PER_DAY) * STEP_S / 3600
            night = not (6 <= hour < 23)
            if ctrl == "schedule":
                Tset = np.full(Z, NIGHT if night else COMFORT)
            elif ctrl in ("reactive", "reactive_standby"):
                Tset = np.full(Z, ECO if ctrl == "reactive" else STANDBY)
                if k_hat == FALL:
                    Tset[last_zone] = COMFORT
                elif ZONE_OF[k_hat] >= 0:
                    Tset[ZONE_OF[k_hat]] = NIGHT if k_hat == SLEEP else COMFORT
            else:
                tout_f = Tout[t:t + H * SLOT:SLOT]
                tout_f = np.pad(tout_f, (0, H - len(tout_f)), mode="edge")
                tout_f = tout_f + rng.normal(0, 1.0)          # forecast error
                # Most-likely intent trajectory (what a greedy agent plans on).
                ml = np.empty((1, H), dtype=np.int64)
                s_ = k_hat
                for j in range(H):
                    h = int(hour + j * DT_SLOT / 3600) % 24
                    s_ = int(hmm.A_look[h][s_].argmax())
                    ml[0, j] = s_
                fut = ml if ctrl == "greedy" else np.vstack(
                    [sample_futures(hmm, posts[t].astype(float), hour, H, N, rng), ml])
                req = required_matrix(fut, last_zone)
                for z in range(Z):
                    base = NIGHT if (z == BED and night) else COMFORT
                    plans = candidate_plans(H, base)
                    P = len(plans)
                    Tt = np.full((P, 1), T[z])
                    E = np.zeros(P)
                    disc = np.zeros((P, fut.shape[0]))
                    for j in range(H):
                        Tt, e = twin.advance(z, Tt, tout_f[j], plans[:, j:j + 1])
                        E += e[:, 0]
                        disc += np.maximum(0, req[z][None, :, j] - Tt) * DT_SLOT / 3600
                    # Agent proposal: cheapest plan that is fine on the
                    # most-likely future.
                    ok_g = disc[:, -1] <= budget
                    i_g = int(np.argmin(np.where(ok_g, E, np.inf))) if ok_g.any() else 0
                    if ctrl == "greedy":
                        Tset[z] = plans[i_g, 0]
                        continue
                    d = disc[:, :-1]
                    if risk == "cvar":
                        q = np.quantile(d, alpha, axis=1, keepdims=True)
                        rk = np.nanmean(np.where(d >= q, d, np.nan), axis=1)
                    else:
                        rk = d.mean(1)
                    ok = rk <= budget
                    decisions += 1
                    if ok[i_g]:
                        i = i_g                      # proposal certified safe
                    else:
                        vetoes += 1                  # shield vetoes and repairs
                        i = int(np.argmin(np.where(ok, E, np.inf))) if ok.any() else 0
                    Tset[z] = plans[i, 0]
        T, q = bld.step(T, Tout[t], Tset)
        energy += q.sum() * STEP_S / 3.6e6
        temps[t] = T
        setp[t] = Tset

    zone = true_zone(home.labels)
    req = np.where(zone >= 0, REQ_LEVEL[home.labels], -np.inf)
    zt = temps[np.arange(T_all), np.maximum(zone, 0)]
    deficit = np.where(zone >= 0, np.maximum(0, req - zt), 0.0)
    occupied = zone >= 0
    days = T_all / STEPS_PER_DAY
    bad = deficit > 0.5
    runs, cnt = 0, 0
    for b in bad:                  # >0.5 K for >= 10 min -> manual override
        cnt = cnt + 1 if b else 0
        if cnt == 20:
            runs += 1
    fall_def = deficit[home.labels == FALL]
    return dict(energy_kwh_day=energy / days,
                discomfort_kh_day=float(deficit.sum() * STEP_S / 3600 / days),
                pct_uncomfortable=float(100 * (deficit[occupied] > 0.5).mean()),
                interventions_day=runs / days,
                fall_max_deficit=float(fall_def.max()) if fall_def.size else 0.0,
                veto_pct=100.0 * vetoes / max(1, decisions),
                temps=temps, setpoints=setp, zone=zone)
