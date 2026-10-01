"""Synthetic multimodal ambient-assisted-living (AAL) home simulator.

Generates ground-truth daily routines (activities of daily living, ADL) for
heterogeneous occupants, injects rare fall emergencies, and produces 11
multimodal sensor streams with first-order dynamics, slow drift, sensor noise
and benign impact artefacts.
"""
from dataclasses import dataclass
import numpy as np

from .config import (STEPS_PER_DAY, STEP_S, K, S, TARGETS, LAG, NOISE, DRIFT,
                     FLOOR, FALL_SPIKE_G, BENIGN_SPIKE_P, BENIGN_SPIKE_G,
                     SLEEP, HYGIENE, COOK, EAT, RELAX, WORK, AWAY, FALL)

MIN = 60 // STEP_S  # steps per minute


@dataclass
class Home:
    labels: np.ndarray      # (T,) int activity labels
    x: np.ndarray           # (T, S) sensor measurements
    hour: np.ndarray        # (T,) hour of day 0..23
    fall_onsets: np.ndarray  # indices of fall onsets
    days: int


def _home_profile(rng):
    return dict(
        wake=rng.normal(6.8, 0.6),
        bed=rng.normal(22.9, 0.6),
        p_away=rng.uniform(0.15, 0.9),      # prob. of leaving for work on weekdays
        fall_rate=rng.uniform(0.15, 0.35),   # falls per day (elevated-risk cohort)
        scale=rng.uniform(0.7, 1.3, size=(K, S)),  # personal/home-specific signature
        bath_rate=rng.uniform(0.3, 0.7),     # extra bathroom visits per awake hour
    )


def _day_schedule(rng, prof, weekday):
    """Return an array of STEPS_PER_DAY activity labels for one day."""
    day = np.full(STEPS_PER_DAY, SLEEP, dtype=np.int64)
    t = int(max(4.5, rng.normal(prof["wake"] + (0 if weekday else 1.0), 0.35)) * 60 * MIN)
    bed = int(min(23.9, rng.normal(prof["bed"], 0.4)) * 60 * MIN)

    def put(act, minutes):
        nonlocal t
        n = max(1, int(minutes * MIN))
        e = min(t + n, bed)
        if e > t:
            day[t:e] = act
        t = e

    put(HYGIENE, rng.uniform(12, 30))
    put(COOK, rng.uniform(8, 20))
    put(EAT, rng.uniform(12, 25))
    goes_out = weekday and rng.random() < prof["p_away"]
    if goes_out:
        put(RELAX, rng.uniform(5, 30))
        ret = rng.normal(17.5, 0.8) * 60 * MIN
        put(AWAY, max(60, (ret - t) / MIN))
    else:
        put(WORK, rng.uniform(90, 180))
        put(HYGIENE, rng.uniform(3, 8))
        put(RELAX, rng.uniform(20, 60))
        put(COOK, rng.uniform(15, 35))
        put(EAT, rng.uniform(15, 30))
        if rng.random() < 0.6:
            put(AWAY, rng.uniform(40, 150))
        put(WORK, rng.uniform(60, 150))
    put(RELAX, rng.uniform(20, 70))
    put(COOK, rng.uniform(20, 45))
    put(EAT, rng.uniform(20, 40))
    put(RELAX, max(10, (bed - t) / MIN - 15))
    put(HYGIENE, rng.uniform(8, 20))
    # Remainder already SLEEP.

    # Short bathroom visits during awake in-home activities.
    awake_home = np.isin(day, [RELAX, WORK, EAT])
    n_visits = rng.poisson(prof["bath_rate"] * awake_home.sum() / (60 * MIN))
    for _ in range(n_visits):
        idx = np.flatnonzero(awake_home)
        if idx.size == 0:
            break
        s0 = rng.choice(idx)
        day[s0:s0 + int(rng.uniform(3, 8) * MIN)] = HYGIENE
    # Night-time bathroom visits.
    for _ in range(rng.poisson(0.8)):
        s0 = int(rng.uniform(0.5, 5.0) * 60 * MIN)
        if day[s0] == SLEEP:
            day[s0:s0 + int(rng.uniform(3, 7) * MIN)] = HYGIENE
    return day


def _inject_falls(rng, labels, prof):
    onsets = []
    days = len(labels) // STEPS_PER_DAY
    for d in range(days):
        if rng.random() >= prof["fall_rate"]:
            continue
        base = d * STEPS_PER_DAY
        cand = np.flatnonzero(np.isin(labels[base:base + STEPS_PER_DAY],
                                      [HYGIENE, COOK, RELAX, WORK, EAT]))
        cand = cand[(cand > 30 * MIN) & (cand < STEPS_PER_DAY - 60 * MIN)]
        if cand.size == 0:
            continue
        s0 = base + rng.choice(cand)
        dur = int(rng.uniform(8, 25) * MIN)
        labels[s0:s0 + dur] = FALL
        # Once help arrives the occupant rests.
        labels[s0 + dur:s0 + dur + int(20 * MIN)] = RELAX
        onsets.append(s0)
    return np.array(onsets, dtype=np.int64)


def simulate_home(seed, days=10):
    rng = np.random.default_rng(seed)
    prof = _home_profile(rng)
    labels = np.concatenate([_day_schedule(rng, prof, weekday=(d % 7) < 5)
                             for d in range(days)])
    fall_onsets = _inject_falls(rng, labels, prof)

    T = len(labels)
    tgt = TARGETS[labels] * prof["scale"][labels]      # (T, S)
    x = np.empty((T, S))
    z = tgt[0].copy()
    drift = np.zeros(S)
    rho = 0.995
    eps = rng.standard_normal((T, S))
    eta = rng.standard_normal((T, S))
    for t in range(T):
        drift = rho * drift + np.sqrt(1 - rho ** 2) * DRIFT * eta[t]
        z = z + LAG * (tgt[t] + drift - z)
        x[t] = np.maximum(FLOOR, z + NOISE * eps[t])
    # Wearable impact transients.
    x[fall_onsets, 9] = FALL_SPIKE_G + 0.3 * rng.standard_normal(len(fall_onsets))
    awake = np.flatnonzero(np.isin(labels, [HYGIENE, COOK, EAT, RELAX, WORK]))
    benign = awake[rng.random(awake.size) < BENIGN_SPIKE_P]
    x[benign, 9] = BENIGN_SPIKE_G + 0.4 * rng.standard_normal(benign.size)
    hour = (np.arange(T) % STEPS_PER_DAY) * STEP_S // 3600
    return Home(labels=labels, x=x, hour=hour.astype(np.int64),
                fall_onsets=fall_onsets, days=days)
