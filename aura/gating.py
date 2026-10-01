"""Edge-node uplink gating policies and the closed edge-gateway loop.

Every policy runs the *same* gateway intent engine; only what the nodes send
(and how the gateway interprets silence) differs.

Policies
--------
oracle    : every sample of every sensor is transmitted (upper bound).
periodic  : every sensor reports every ``period`` steps (duty-cycled IoT).
sod       : send-on-delta with a fixed dead-band, zero-order-hold twin.
dps       : dual-prediction scheme with a damped-trend twin, fixed dead-band.
aura      : AURA - intent-uncertainty-coupled dead-bands broadcast by the
            gateway as a quantised attention vector, silence-is-informative
            interval likelihood, and a safety floor on the emergency intent.

All policies forward wearable impact transients immediately (event bypass).
"""
import numpy as np

from .config import (K, S, NOISE, E_TX_MJ, E_RX_BEACON_MJ, E_SENSE_MJ,
                     BATTERY_J, P_SLEEP_UW, STEP_S)
from .intent import IMPACT_G, ACCEL


def default_policies():
    return {
        "Oracle (stream all)": dict(kind="oracle"),
        "Periodic (2 min)": dict(kind="periodic", period=4),
        "Send-on-Delta": dict(kind="sod", c=2.0),
        "Dual-Prediction": dict(kind="dps", c=2.0),
        "AURA (proposed)": dict(kind="aura"),
    }


AURA_DEFAULTS = dict(c_min=1.0, c_max=4.0, kappa=3.0, safety_floor=0.10,
                     levels=6, loosen_hold=30, interval=True, adaptive=True,
                     lookahead=True)


def run_policy(home, hmm, pol, keep_theta=False):
    pol = dict(pol)
    kind = pol["kind"]
    if kind == "aura":
        pol = {**AURA_DEFAULTS, **pol}
    interval = pol.get("interval", False)
    trend = kind == "dps" or pol.get("predictor") == "trend"
    adaptive = kind == "aura" and pol.get("adaptive", True)

    X, hours = home.x, home.hour
    T = len(X)
    post = hmm.prior.copy()
    posts = np.empty((T, K), dtype=np.float32)
    tx = np.zeros(S, dtype=np.int64)
    events = 0
    beacons = 0
    theta_trace = np.empty((T, S), dtype=np.float32) if keep_theta else None

    # Shared twin state (identical at node and gateway).
    ref = X[0].copy()            # last transmitted value
    slope = np.zeros(S)
    t_ref = np.zeros(S)
    tx += 1

    if kind in ("sod", "dps"):
        theta = pol["c"] * NOISE
    elif kind == "aura":
        lv = pol["c_min"] * (pol["c_max"] / pol["c_min"]) ** (
            np.arange(pol["levels"]) / (pol["levels"] - 1))
        if not adaptive:
            lv = np.array([pol.get("c_fixed", 2.0)])
        level = np.full(S, pol["levels"] - 1)
        if not adaptive:
            level[:] = 0
        theta = lv[level] * NOISE
        last_beacon = -10 ** 9
    else:
        theta = np.zeros(S)

    for t in range(T):
        x = X[t]
        impact = x[ACCEL] > IMPACT_G
        if impact:
            events += 1
        skip = np.zeros(S, dtype=bool)
        skip[ACCEL] = impact
        xm = x.copy()
        if impact:
            xm[ACCEL] = ref[ACCEL]     # transient is not fed to the twin

        if kind == "oracle":
            sent = np.ones(S, dtype=bool)
            xhat = xm
            if t:
                tx += 1
        elif kind == "periodic":
            sent = np.full(S, t % pol["period"] == 0)
            if sent[0] and t:
                ref = xm.copy()
                tx += 1
            xhat = ref
        else:
            if trend:
                pred = np.maximum(0, ref + slope * np.minimum(t - t_ref, 20))
            else:
                pred = ref
            sent = np.abs(xm - pred) > theta
            if t == 0:
                sent[:] = False
            if sent.any():
                tx += sent
                if trend:
                    dt = np.maximum(t - t_ref, 1)
                    new_slope = (xm - ref) / dt
                    slope = np.where(sent, 0.5 * slope + 0.5 * new_slope, slope)
                ref = np.where(sent, xm, ref)
                t_ref = np.where(sent, t, t_ref)
                pred = np.where(sent, xm, pred)
            xhat = pred

        ll = hmm.loglik(xhat, sent, theta, interval, skip=skip)
        post = hmm.step(post, hours[t], ll, impact)
        posts[t] = post

        if adaptive:
            D = hmm.discriminability(post, hours[t], pol["safety_floor"],
                                     pol["lookahead"])
            c_star = pol["c_min"] + (pol["c_max"] - pol["c_min"]) * np.exp(-pol["kappa"] * D)
            want = np.argmin(np.abs(np.log(lv)[None, :] - np.log(c_star)[:, None]), axis=1)
            tighten = want < level
            loosen = want > level
            if tighten.any() or (loosen.any() and t - last_beacon >= pol["loosen_hold"]):
                level = np.where(tighten | (loosen & (t - last_beacon >= pol["loosen_hold"])),
                                 want, level)
                theta = lv[level] * NOISE
                beacons += 1
                last_beacon = t
        if keep_theta:
            theta_trace[t] = theta / NOISE

    steps = T
    # Per-node energy (each sensor is an independent battery node).
    e_node_mj = steps * E_SENSE_MJ + tx * E_TX_MJ + beacons * E_RX_BEACON_MJ
    e_node_mj[ACCEL] += events * E_TX_MJ
    power_w = e_node_mj * 1e-3 / (steps * STEP_S) + P_SLEEP_UW * 1e-6
    life_days = BATTERY_J / power_w / 86400
    return dict(posts=posts, tx=tx, events=events, beacons=beacons,
                tx_rate=float(tx.sum() + events) / (steps * S),
                life_days=life_days, e_node_mj=e_node_mj,
                theta_trace=theta_trace, steps=steps)
