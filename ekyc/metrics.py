"""
Metrics. Everything works on logits (higher = more likely fake), and a frame is called fake iff logit > threshold.

  AAR (attack acceptance rate)   = share of fake frames accepted as genuine  = 1 - recall on fakes
  BPCER (genuine rejection rate) = share of real frames rejected as fake    = 1 - specificity

The threshold is fixed on FF++ validation real frames so that 1% of them are rejected (ISO/IEC 30107-3 style).
"""
from fractions import Fraction

import numpy as np

from ekyc.config import TARGET_BPCER

AUC_TIE = 1e-3  # model selection: val AUCs closer than this count as a tie (then the lower balanced val loss wins)


def threshold_at_bpcer(real, target=TARGET_BPCER, weights=None):
    """The smallest real score t such that the (weighted) share of real scores strictly above t is <= target.
    Exact integer arithmetic for integer weights. Unweighted with n = 4,477: t is the 4,433rd smallest score and 44
    frames (0.98%) lie above it. Frames equal to t count as genuine."""
    vals, inv = np.unique(np.asarray(real), return_inverse=True)
    w = np.ones(len(inv), dtype=np.int64) if weights is None else np.asarray(weights)
    per_value = np.bincount(inv, weights=w, minlength=len(vals))
    if np.issubdtype(w.dtype, np.integer):
        per_value = np.rint(per_value).astype(np.int64)
    total = per_value.sum()
    above = total - np.cumsum(per_value)
    frac = Fraction(target).limit_denominator(10**6)
    ok = above * frac.denominator <= frac.numerator * total
    return float(vals[np.argmax(ok)])


def aar(fake, t):
    return float(np.mean(np.asarray(fake) <= t))


def bpcer(real, t):
    return float(np.mean(np.asarray(real) > t))


def auc(scores, labels, weights=None):
    """Area under the ROC curve (fake = positive), ties counted as one half; equals sklearn's roc_auc_score,
    including with sample weights."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    s, y, w = s[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]  # tie groups
    pos = np.add.reduceat(np.where(y, w, 0.0), starts)
    neg = np.add.reduceat(np.where(y, 0.0, w), starts)
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def bce(logits, labels):
    """Per-frame binary cross-entropy from logits."""
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    return np.where(y == 1, np.logaddexp(0.0, -z), np.logaddexp(0.0, z))


def balanced_bce(logits, labels, methods):
    """0.5 x mean loss on real frames + 0.5 x the mean over fake methods of each method's mean loss, so the selection
    loss doesn't depend on how many frames each method happens to have."""
    loss, y, m = bce(logits, labels), np.asarray(labels), np.asarray(methods)
    fake_methods = sorted(set(m[y == 1]))
    return float(0.5 * loss[y == 0].mean() + 0.5 * np.mean([loss[m == k].mean() for k in fake_methods]))


def confusion_metrics(scores, labels, t):
    """Fake = positive class: recall = sensitivity = 1 - AAR, specificity = 1 - BPCER."""
    pred, y = np.asarray(scores) > t, np.asarray(labels) == 1
    tp, fn = int(np.sum(pred & y)), int(np.sum(~pred & y))
    fp, tn = int(np.sum(pred & ~y)), int(np.sum(~pred & ~y))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "accuracy": (tp + tn) / (tp + fn + fp + tn),
            "precision": prec, "recall": rec, "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else float("nan")}


def select(records, key_auc="val_auc", key_loss="val_loss_bal"):
    """The selection rule, for epochs and for learning rates: among records whose val AUC is within
    AUC_TIE of the best, take the lowest balanced val loss; records are assumed ordered so that earlier = preferred
    on a remaining tie (earlier epoch, lower learning rate). Returns the chosen index."""
    best = max(r[key_auc] for r in records)
    cands = [i for i, r in enumerate(records) if best - r[key_auc] < AUC_TIE]
    return min(cands, key=lambda i: (records[i][key_loss], i))
