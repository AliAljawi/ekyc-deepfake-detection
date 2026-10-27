"""
Cluster bootstrap

A replicate redraws each stratums clusters (source clips, images or identities) with replacement, and every frame
gets its clusters draw count as its weight. Strata are drawn independently, every quantity that uses a stratum uses
the same draws (they depend only on the stratum and the chunk number), so differences between models are paired, and
the first 2,000 replicates of a 10,000-replicate analysis are the same draws as a 2,000-replicate one.
"""
from fractions import Fraction

import numpy as np

SEED = 20261022
STRATA = {"CDF": 0, "FF": 1, "SD": 2, "DIT": 3, "VAL": 4, "ID_CDF": 5, "ID_FF": 6, "ID_DF40": 7}
CHUNK = 250


def weights(stratum, n_clusters, B, chunk=CHUNK):
    """Yields replicate weight matrices (b, n_clusters), B rows in total: each row is one multinomial draw of
    n_clusters clusters with equal probabilities."""
    s = STRATA[stratum]
    for c, start in enumerate(range(0, B, chunk)):
        b = min(chunk, B - start)
        idx = np.random.default_rng([SEED, s, c]).integers(0, n_clusters, (b, n_clusters))
        flat = (idx + n_clusters * np.arange(b)[:, None]).ravel()
        yield np.bincount(flat, minlength=b * n_clusters).reshape(b, n_clusters).astype(np.float64)


def all_weights(stratum, n_clusters, B):
    return np.concatenate(list(weights(stratum, n_clusters, B)))


def cluster_sums(codes, n_clusters, values):
    """Frames per cluster (C,) and per-cluster sums of each column of values (n_frames, K) -> (C, K)."""
    n = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(len(codes), -1)
    a = np.stack([np.bincount(codes, weights=values[:, k], minlength=n_clusters) for k in range(values.shape[1])], 1)
    return n, a


def rate_reps(W, n, a):
    """Replicate rates (B, K): weighted share of frames with value 1 (ratio estimator over clusters)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ a) / (W @ n)[:, None]


class AUCReps:
    """Replicate AUCs for fixed scores: sort once, then each replicate only re-weights the frames."""

    def __init__(self, scores, is_pos):
        self.order = np.argsort(np.asarray(scores), kind="stable")
        s = np.asarray(scores)[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
        self.pos = np.asarray(is_pos, dtype=bool)[self.order]

    def __call__(self, frame_w):
        """frame_w: (b, n_frames) weights in the original frame order -> (b,) AUCs."""
        w = frame_w[:, self.order]
        gp = np.add.reduceat(np.where(self.pos, w, 0.0), self.starts, axis=1)
        gn = np.add.reduceat(np.where(self.pos, 0.0, w), self.starts, axis=1)
        below = np.cumsum(gn, axis=1) - gn
        with np.errstate(invalid="ignore", divide="ignore"):
            return (gp * (below + 0.5 * gn)).sum(1) / (gp.sum(1) * gn.sum(1))


class ThresholdReps:
    """Replicate thresholds with the rule: the smallest real score with at most `target` of the
    (weighted) real frames strictly above it."""

    def __init__(self, real_scores, target):
        self.vals, inv = np.unique(np.asarray(real_scores), return_inverse=True)
        self.order = np.argsort(inv, kind="stable")
        self.starts = np.r_[0, np.flatnonzero(np.diff(inv[self.order])) + 1]
        self.frac = Fraction(target).limit_denominator(10**6)

    def __call__(self, frame_w):
        per_value = np.add.reduceat(frame_w[:, self.order], self.starts, axis=1)
        total = per_value.sum(1, keepdims=True)
        above = total - np.cumsum(per_value, axis=1)
        ok = above * self.frac.denominator <= self.frac.numerator * total + 1e-9
        return self.vals[np.argmax(ok, axis=1)]


class AcceptReps:
    """Replicate share of frames accepted (score <= a per-replicate threshold)."""

    def __init__(self, scores):
        self.order = np.argsort(np.asarray(scores), kind="stable")
        self.sorted = np.asarray(scores)[self.order]

    def __call__(self, frame_w, thresholds):
        w = np.cumsum(frame_w[:, self.order], axis=1)
        pos = np.searchsorted(self.sorted, thresholds, side="right")
        accepted = np.where(pos > 0, w[np.arange(len(w)), np.maximum(pos - 1, 0)], 0.0)
        return accepted / w[:, -1]


def ci(reps, level=0.95):
    """Percentile CI over replicates (ignoring the rare replicate where a quantity is undefined)."""
    r = np.asarray(reps, dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return float("nan"), float("nan")
    lo, hi = np.percentile(r, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def p_two_sided(reps):
    d = np.asarray(reps)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min(1 + np.sum(d >= 0), 1 + np.sum(d <= 0)) / (len(d) + 1)))


def p_one_sided_below_zero(reps):
    """H2: p-value for 'the true value is < 0'."""
    d = np.asarray(reps)
    d = d[np.isfinite(d)]
    return float((1 + np.sum(d >= 0)) / (len(d) + 1))


def holm(pvals):
    """Holm (1979) step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()


def mcnemar(b, c):
    """Exact two-sided McNemar test on b and c discordant units (p = 1 if there are none)."""
    from scipy.stats import binomtest
    return float(binomtest(int(b), int(b + c), 0.5).pvalue) if b + c else 1.0
