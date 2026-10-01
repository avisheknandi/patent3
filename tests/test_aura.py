"""Fast unit/integration tests for the AURA reference implementation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura.config import K, S, FALL, NOISE                   # noqa: E402
from aura.simulator import simulate_home                     # noqa: E402
from aura.intent import IntentHMM                            # noqa: E402
from aura.gating import run_policy                           # noqa: E402
from aura.metrics import recognition                         # noqa: E402
from aura.twin import run_controller                         # noqa: E402

HMM = IntentHMM().fit([simulate_home(s, 4) for s in range(3)])
HOME = simulate_home(99, 2)


def test_simulator_shapes():
    assert HOME.x.shape == (len(HOME.labels), S)
    assert HOME.labels.min() >= 0 and HOME.labels.max() < K


def test_transition_matrices_are_stochastic():
    assert np.allclose(HMM.A.sum(2), 1)
    assert np.allclose(HMM.A_look.sum(2), 1)


def test_interval_likelihood_widens_with_deadband():
    """Silence over a wider band must be less informative (flatter)."""
    x = HMM.mu[0]
    sent = np.zeros(S, dtype=bool)
    narrow = HMM.loglik(x, sent, 0.5 * NOISE, True)
    wide = HMM.loglik(x, sent, 20 * NOISE, True)
    assert np.ptp(wide) < np.ptp(narrow)


def test_discriminability_zero_when_certain_without_floor():
    p = np.zeros(K)
    p[0] = 1.0
    d = HMM.discriminability(p, 3, safety_floor=0.0, lookahead=False)
    assert np.allclose(d, 0)


def test_aura_saves_traffic_vs_oracle_and_keeps_accuracy():
    ro = run_policy(HOME, HMM, dict(kind="oracle"))
    ra = run_policy(HOME, HMM, dict(kind="aura"))
    assert ra["tx_rate"] < 0.25 * ro["tx_rate"]
    f_o = recognition(HOME.labels, ro["posts"])["macro_f1"]
    f_a = recognition(HOME.labels, ra["posts"])["macro_f1"]
    assert f_a > f_o - 0.1
    assert np.allclose(ra["posts"].sum(1), 1, atol=1e-4)


def test_shield_runs_and_reports():
    posts = run_policy(HOME, HMM, dict(kind="aura"))["posts"]
    out = run_controller(HOME, posts, HMM, "shield", seed=1, N=16, H=12)
    assert out["energy_kwh_day"] > 0
    assert 0 <= out["veto_pct"] <= 100
