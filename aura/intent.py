"""Gateway-side probabilistic intent (activity) engine.

A time-of-day-dependent hidden Markov model whose forward filter consumes the
*gated* sensor stream. Two observation models are supported per sensor/step:

* ``point``    - the gateway treats its reconstructed value as if measured
                 (conventional event-based reporting).
* ``interval`` - silence-is-informative likelihood: when a node stays silent
                 the gateway knows the true value lies inside the dead-band
                 ``[x_hat - theta, x_hat + theta]`` and integrates the class
                 likelihood over that band (AURA, claim element C2).

Impact transients from the wearable are modelled as a *transition-conditioned*
observation (onset of FALL vs benign impacts during steady activities).
"""
import numpy as np
from scipy.special import ndtr

from .config import K, S, FALL, STEPS_PER_DAY

IMPACT_G = 1.5          # gateway impact detector on the wearable channel
ACCEL = 9
LOG_FLOOR = -30.0


class IntentHMM:
    def __init__(self, beta=0.35, backoff=20.0, lookahead=10):
        self.beta = beta          # likelihood tempering (temporal autocorrelation)
        self.backoff = backoff
        self.lookahead = lookahead

    # ------------------------------------------------------------------ fit
    def fit(self, homes):
        X = np.concatenate([h.x for h in homes])
        y = np.concatenate([h.labels for h in homes])
        hr = np.concatenate([h.hour for h in homes])
        impact = X[:, ACCEL] > IMPACT_G
        self.mu = np.zeros((K, S))
        self.sd = np.zeros((K, S))
        for k in range(K):
            m = (y == k) & ~impact
            self.mu[k] = X[m].mean(0)
            self.sd[k] = X[m].std(0) + 1e-3
        self.sd = np.maximum(self.sd, 0.05 * self.sd.max(0))

        # Hour-dependent transitions with back-off to the pooled matrix.
        prev = np.concatenate([h.labels[:-1] for h in homes])
        nxt = np.concatenate([h.labels[1:] for h in homes])
        hh = np.concatenate([h.hour[1:] for h in homes])
        C = np.zeros((24, K, K))
        np.add.at(C, (hh, prev, nxt), 1.0)
        G = C.sum(0) + 0.5 * (1 - np.eye(K)) + 1.0 * np.eye(K)
        G /= G.sum(1, keepdims=True)
        A = C + self.backoff * G[None]
        A /= A.sum(2, keepdims=True)
        self.A = A

        # Transition-conditioned impact model.
        onset = (nxt == FALL) & (prev != FALL)
        imp_n = np.concatenate([h.x[1:, ACCEL] > IMPACT_G for h in homes])
        p_onset = (imp_n[onset].sum() + 1) / (onset.sum() + 2)
        steady = ~onset
        p_benign = np.array([(imp_n[steady & (nxt == k)].sum() + 0.1) /
                             ((steady & (nxt == k)).sum() + 1) for k in range(K)])
        M_imp = np.tile(p_benign[None, :], (K, 1))
        M_imp[np.arange(K) != FALL, FALL] = p_onset
        self.A_imp = A * M_imp[None]
        self.A_noimp = A * (1 - M_imp)[None]
        self.A_look = np.stack([np.linalg.matrix_power(A[h], self.lookahead)
                                for h in range(24)])
        self.prior = np.bincount(y, minlength=K) / len(y)
        return self

    # --------------------------------------------------------------- filter
    def loglik(self, xhat, sent, theta, interval, skip=None):
        """Per-class log-likelihood of one step's gated observation."""
        mu, sd = self.mu, self.sd
        z = (xhat[None, :] - mu) / sd
        ll = -0.5 * z ** 2 - np.log(sd)                         # point model
        if interval:
            silent = ~sent
            if silent.any():
                th = theta[silent][None, :]
                s_ = sd[:, silent]
                hi = ndtr((xhat[silent][None, :] + th - mu[:, silent]) / s_)
                lo = ndtr((xhat[silent][None, :] - th - mu[:, silent]) / s_)
                ll[:, silent] = np.log(np.maximum(hi - lo, 1e-13))
        ll = np.maximum(ll, LOG_FLOOR)
        if skip is not None:
            ll[:, skip] = 0.0
        return self.beta * ll.sum(1)

    def step(self, post, hour, ll, impact):
        T = self.A_imp[hour] if impact else self.A_noimp[hour]
        pred = post @ T
        lp = np.log(np.maximum(pred, 1e-300)) + ll
        lp -= lp.max()
        p = np.exp(lp)
        return p / p.sum()

    # ---------------------------------------------- AURA attention vector
    def discriminability(self, post, hour, safety_floor, lookahead=True):
        """Per-sensor expected between-intent separability under the
        look-ahead intent distribution with a safety floor on FALL."""
        q = post @ self.A_look[hour] if lookahead else post.copy()
        q = (1 - safety_floor) * q
        q[FALL] += safety_floor
        mbar = q @ self.mu
        return (q[:, None] * ((self.mu - mbar[None]) / self.sd) ** 2).sum(0)
