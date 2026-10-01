"""Evaluation metrics for intent recognition and emergency detection."""
import numpy as np
from sklearn.metrics import f1_score

from .config import FALL, STEP_S, STEPS_PER_DAY, K


def recognition(labels, posts):
    pred = posts.argmax(1)
    return dict(acc=float((pred == labels).mean()),
                macro_f1=float(f1_score(labels, pred, average="macro",
                                        labels=list(range(K)), zero_division=0)))


def fall_detection(labels, posts, onsets, tau=0.5):
    """Recall, mean/95th-pct detection latency (s) and false alarms per day."""
    alarm = posts[:, FALL] > tau
    T = len(labels)
    delays, hit = [], 0
    in_event = np.zeros(T, dtype=bool)
    for s in onsets:
        e = s
        while e < T and labels[e] == FALL:
            e += 1
        in_event[s:e] = True
        idx = np.flatnonzero(alarm[s:e])
        if idx.size:
            hit += 1
            delays.append(int(idx[0]) * STEP_S)
    rising = np.flatnonzero(alarm & ~np.r_[False, alarm[:-1]])
    false_alarms = int((~in_event[rising]).sum())
    days = T / STEPS_PER_DAY
    return dict(recall=hit / max(1, len(onsets)),
                delay_s=float(np.mean(delays)) if delays else float("nan"),
                delays=delays,
                fa_per_day=false_alarms / days)
