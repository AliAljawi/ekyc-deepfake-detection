"""
analyze.py - step 5: every reported number, computed from the stored scores with the rules (thresholds, H1 with its reading rule and McNemar's test, H2, RQ3, fairness, threshold trade-off and
uncertainty, seeds, the DeepfakeBench sanity check, standard metrics).

Reads outputs/scores/<run>/scores.npz (from evaluate.py) and the master list. Writes tables to outputs/results/
(numbers only, no images) and a readable summary, outputs/results/RESULTS.md. This is the only place thresholds are
computed (export_demo.py reads them from thresholds.csv).

Usage (VS Code terminal, inside the code folder):
    .venv\\Scripts\\python.exe -B analyze.py                     # full analysis (~5 min)
    .venv\\Scripts\\python.exe -B analyze.py --B 500             # quick look with fewer bootstrap replicates
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import LEVELS, OUT, RESULTS, TARGET_BPCER  # noqa: E402
from ekyc.utils import atomic_write, git_commit, setup_console, setup_logging  # noqa: E402

VIT, CNN, PROBE, ZS = "clip_lora", "effnet_b4", "clip_probe", "clip_zeroshot"
NAMES = {VIT: "ViT (CLIP + LoRA)", CNN: "EfficientNet-B4", PROBE: "Frozen CLIP + probe", ZS: "Zero-shot CLIP"}
TYPES = {"faceswap": ("inswap", "simswap"), "reenactment": ("fomm", "sadtalker"), "diffusion": ("sd2.1", "DiT")}
DF40 = tuple(m for ms in TYPES.values() for m in ms)
GROUPS = {  # name: (set, method), stratum of its bootstrap clusters
    "ffpp_real": (("ffpp_test", "real"), "FF"), "ffpp_DF": (("ffpp_test", "DF"), "FF"),
    "ffpp_F2F": (("ffpp_test", "F2F"), "FF"), "ffpp_FS": (("ffpp_test", "FS"), "FF"),
    "ffpp_NT": (("ffpp_test", "NT"), "FF"),
    "cdf_real": (("cdf_test", "real"), "CDF"), "cdf_fake": (("cdf_test", "Celeb-synthesis"), "CDF"),
    "inswap": (("df40_test", "inswap"), "CDF"), "simswap": (("df40_test", "simswap"), "CDF"),
    "fomm": (("df40_test", "fomm"), "CDF"), "sadtalker": (("df40_test", "sadtalker"), "CDF"),
    "sd2.1": (("df40_test", "sd2.1"), "SD"), "DiT": (("df40_test", "DiT"), "DIT"),
}
REAL_OF = {g: ("ffpp_real" if g.startswith("ffpp") else "cdf_real") for g in GROUPS if not g.endswith("_real")}
TRADEOFF = (0.005, 0.01, 0.02, 0.05)
SANITY_REF = 0.7487


def sanity_band(a):
    if 0.65 <= a <= 0.80:
        return "consistent with 0.7487"
    if 0.60 <= a < 0.65:
        return "lower than expected"
    if 0.80 < a <= 0.85:
        return "higher than expected (run a leakage check)"
    return "outside the bands: run the checks that don't use test scores"


class Data:
    """The scores of every run, aligned on the same test frames, plus the master-list columns they need."""

    def __init__(self, scores_dir, log):
        import numpy as np
        from ekyc.data import index_sha256, load_index
        sha = index_sha256()
        idx = load_index(columns=["set", "method", "label", "video", "cluster", "identity", "gender", "age"])
        self.runs = {}
        for d in sorted(p for p in Path(scores_dir).iterdir() if (p / "done.json").exists()):
            z = np.load(d / "scores.npz")
            info = json.loads((d / "scores.json").read_text(encoding="utf-8"))
            if str(z["index_sha256"]) != sha:
                raise AssertionError(f"{d.name}: scored on a different master list")
            self.runs[d.name] = {"kind": info["kind"], "seed": info.get("seed"), "row": z["row"],
                                 "logit": z["logit"], "val_row": z["val_row"], "val_logit": z["val_logit"],
                                 "info": info}
        if not self.runs:
            raise FileNotFoundError(f"no finished score folders in {scores_dir}")
        first = next(iter(self.runs.values()))
        for name, r in self.runs.items():
            if not (np.array_equal(r["row"], first["row"]) and np.array_equal(r["val_row"], first["val_row"])):
                raise AssertionError(f"{name}: scored a different set of frames")
            if list(r["info"]["levels"]) != list(LEVELS):
                raise AssertionError(f"{name}: unexpected JPEG levels {r['info']['levels']}")
        self.test = idx.loc[first["row"]].reset_index(drop=True)
        self.val = idx.loc[first["val_row"]].reset_index(drop=True)
        self.limited = any(r["info"].get("limited") for r in self.runs.values())
        self.seeds = sorted({r["seed"] for r in self.runs.values() if r["seed"] is not None})
        self.pos = {g: np.flatnonzero(((self.test["set"] == s) & (self.test["method"] == m)).to_numpy())
                    for g, ((s, m), _) in GROUPS.items()}
        self.val_real = (self.val["label"] == 0).to_numpy()
        # Bootstrap clusters: one code per frame within its stratum
        self.strata = {}
        for stratum in ("CDF", "FF", "SD", "DIT"):
            gs = [g for g, (_, s) in GROUPS.items() if s == stratum]
            names = sorted(set().union(*[set(self.test["cluster"].to_numpy()[self.pos[g]]) for g in gs]))
            self.strata[stratum] = {n: i for i, n in enumerate(names)}
        self.codes = {g: np.array([self.strata[s][c] for c in self.test["cluster"].to_numpy()[self.pos[g]]])
                      for g, (_, s) in GROUPS.items()}
        vids = self.val["video"].to_numpy()[self.val_real]
        self.val_names = sorted(set(vids))
        lookup = {v: i for i, v in enumerate(self.val_names)}
        self.val_codes = np.array([lookup[v] for v in vids])
        log.info(f"Loaded {len(self.runs)} runs: {', '.join(self.runs)}; {len(self.test):,} test frames, "
                 f"{len(self.val):,} val frames" + (" (LIMITED test run)" if self.limited else ""))
        log.info("Clusters: " + ", ".join(f"{s} {len(c):,}" for s, c in self.strata.items())
                 + f", VAL {len(self.val_names)} videos")
        if not self.limited:  # the registered units
            expected = {"FF": 70, "SD": 28_174, "DIT": 28_174}
            got = {s: len(self.strata[s]) for s in expected}
            if got != expected or len(self.val_names) != 140:
                raise AssertionError(f"bootstrap units {got}, VAL {len(self.val_names)}; expected {expected}, VAL 140")

    def run(self, kind, seed=0):
        return self.runs[ZS if kind == ZS else f"{kind}_s{seed}"]

    def logits(self, run, group, level=0):
        return run["logit"][self.pos[group], level]

    def auc_vs_real(self, run, method, level=0):
        """AUC of a fake group against its real reference (Celeb-DF test reals for Celeb-DF and DF40)."""
        import numpy as np
        from ekyc.metrics import auc
        real, fake = self.logits(run, REAL_OF[method], level), self.logits(run, method, level)
        return auc(np.r_[real, fake], np.r_[np.zeros(len(real)), np.ones(len(fake))])

    def unseen_auc(self, run, level=0):
        return sum(type_value({m: self.auc_vs_real(run, m, level) for m in TYPES[k]}, k) for k in TYPES) / 3

    def threshold(self, run, target=TARGET_BPCER):
        from ekyc.metrics import threshold_at_bpcer
        return threshold_at_bpcer(run["val_logit"][self.val_real], target)


def point_tables(D):
    """Thresholds, every group's AAR/BPCER/AUC per run and level, and the standard metrics."""
    import numpy as np
    import pandas as pd
    from ekyc.metrics import aar, auc, bpcer, confusion_metrics
    thr, rows, conf = [], [], []
    for name, r in D.runs.items():
        vr = r["val_logit"][D.val_real]
        for target in TRADEOFF:
            t = D.threshold(r, target)
            above = vr > t
            thr.append({"run": name, "kind": r["kind"], "seed": r["seed"], "target_bpcer": target, "threshold": t,
                        "val_real_frames": int(len(vr)), "val_real_above": int(above.sum()),
                        "val_videos_above": int(D.val.loc[D.val_real].loc[above, "video"].nunique()
                                                if len(vr) else 0)})
        t = D.threshold(r)
        for li, level in enumerate(LEVELS):
            for g in GROUPS:
                x = D.logits(r, g, li)
                if g.endswith("_real"):
                    rows.append({"run": name, "kind": r["kind"], "seed": r["seed"], "level": level, "group": g,
                                 "metric": "BPCER", "value": bpcer(x, t), "frames": len(x)})
                else:
                    xr = D.logits(r, REAL_OF[g], li)
                    rows += [{"run": name, "kind": r["kind"], "seed": r["seed"], "level": level, "group": g,
                              "metric": "AAR", "value": aar(x, t), "frames": len(x)},
                             {"run": name, "kind": r["kind"], "seed": r["seed"], "level": level, "group": g,
                              "metric": "AUC", "value": auc(np.r_[xr, x], np.r_[np.zeros(len(xr)), np.ones(len(x))]),
                              "frames": len(x) + len(xr)}]
            for test_set, gs in (("FF++ test", [g for g in GROUPS if g.startswith("ffpp")]),
                                 ("Celeb-DF test", ["cdf_real", "cdf_fake"])):
                x = np.concatenate([D.logits(r, g, li) for g in gs])
                y = np.concatenate([np.full(len(D.pos[g]), 0 if g.endswith("_real") else 1) for g in gs])
                conf.append({"run": name, "kind": r["kind"], "seed": r["seed"], "level": level, "test_set": test_set,
                             "auc": auc(x, y), **confusion_metrics(x, y, t)})
    return pd.DataFrame(thr), pd.DataFrame(rows), pd.DataFrame(conf)


def type_value(per_method, kind):
    """Type-level value = mean of its two methods' values (dict method -> array or float)."""
    a, b = TYPES[kind]
    return 0.5 * (per_method[a] + per_method[b])


class Boot:
    """Replicate rates for the primary cells: (run, level) columns per group, B replicates, paired across runs."""

    def __init__(self, D, runs, B, log):
        import numpy as np
        from ekyc.bootstrap import cluster_sums, rate_reps, weights
        self.D, self.runs, self.B = D, runs, B
        self.cols = [(name, li) for name in runs for li in range(len(LEVELS))]
        thr = {name: D.threshold(D.runs[name]) for name in runs}
        self.reps = {}
        t0 = time.time()
        for stratum, clusters in D.strata.items():
            gs = [g for g, (_, s) in GROUPS.items() if s == stratum]
            sums = {}
            for g in gs:
                err = np.stack([(D.logits(D.runs[n], g, li) > thr[n]) if g.endswith("_real")
                                else (D.logits(D.runs[n], g, li) <= thr[n]) for n, li in self.cols], 1)
                sums[g] = cluster_sums(D.codes[g], len(clusters), err)
            parts = {g: [] for g in gs}
            for W in weights(stratum, len(clusters), B):
                for g in gs:
                    parts[g].append(rate_reps(W, *sums[g]))
            for g in gs:
                self.reps[g] = np.concatenate(parts[g])
        log.info(f"       bootstrap: {B:,} replicates of every rate for {len(runs)} runs x {len(LEVELS)} levels "
                 f"in {time.time() - t0:.0f} s")

    def rate(self, group, name, level=0):
        return self.reps[group][:, self.cols.index((name, level))]

    def type_aar(self, name, kind, level=0):
        return type_value({m: self.rate(m, name, level) for m in TYPES[kind]}, kind)

    def unseen_aar(self, name, level=0):
        return sum(self.type_aar(name, k, level) for k in TYPES) / 3


class SlowBoot:
    """Replicates of quantities that need per-replicate re-sorting or re-thresholding (B = 2,000): AUCs, the oracle
    threshold on Celeb-DF real frames, and the threshold re-estimated on FF++ val real videos."""

    def __init__(self, D, auc_cells, oracle_runs, sens_runs, B, log):
        import numpy as np
        from ekyc.bootstrap import AcceptReps, AUCReps, ThresholdReps, weights
        t0 = time.time()
        self.auc = {cell: [] for cell in auc_cells}            # (name, level, method) -> replicate AUCs
        self.oracle = {(n, m): [] for n in oracle_runs for m in DF40}
        self.sens = {(n, m): [] for n in sens_runs for m in DF40}
        real = "cdf_real"
        auc_obj = {(n, li, m): AUCReps(np.r_[D.logits(D.runs[n], real, li), D.logits(D.runs[n], m, li)],
                                       np.r_[np.zeros(len(D.pos[real])), np.ones(len(D.pos[m]))])
                   for n, li, m in auc_cells}
        thr_cdf = {n: ThresholdReps(D.logits(D.runs[n], real), TARGET_BPCER) for n in oracle_runs}
        thr_val = {n: ThresholdReps(D.runs[n]["val_logit"][D.val_real], TARGET_BPCER) for n in sens_runs}
        acc = {(n, m): AcceptReps(D.logits(D.runs[n], m)) for n in set(oracle_runs) | set(sens_runs) for m in DF40}
        gens = {s: weights(s, len(c), B) for s, c in D.strata.items()}
        gens["VAL"] = weights("VAL", len(D.val_names), B)
        stratum_of = {g: s for g, (_, s) in GROUPS.items()}
        for _ in range(-(-B // 250)):
            W = {s: next(g) for s, g in gens.items()}
            fw = {g: W[stratum_of[g]][:, D.codes[g]] for g in (real, *DF40)}
            for (n, li, m), obj in auc_obj.items():
                self.auc[(n, li, m)].append(obj(np.concatenate([fw[real], fw[m]], axis=1)))
            for n in oracle_runs:
                t = thr_cdf[n](fw[real])
                for m in DF40:
                    self.oracle[(n, m)].append(acc[(n, m)](fw[m], t))
            if D.val_codes is not None:
                wv = W["VAL"][:, D.val_codes]
                for n in sens_runs:
                    t = thr_val[n](wv)
                    for m in DF40:
                        self.sens[(n, m)].append(acc[(n, m)](fw[m], t))
        for store in (self.auc, self.oracle, self.sens):
            for k in store:
                store[k] = np.concatenate(store[k]) if store[k] else np.full(B, np.nan)
        log.info(f"       bootstrap: {B:,} replicates of AUCs, oracle and re-estimated thresholds in "
                 f"{time.time() - t0:.0f} s")

    def type_auc(self, name, kind, level=0):
        return type_value({m: self.auc[(name, level, m)] for m in TYPES[kind]}, kind)

    def unseen_auc(self, name, level=0):
        return sum(self.type_auc(name, k, level) for k in TYPES) / 3


def summarise(reps, point):
    from ekyc.bootstrap import ci, p_two_sided
    lo, hi = ci(reps)
    return {"estimate": float(point), "ci_low": lo, "ci_high": hi, "p": p_two_sided(reps)}


def verdict(s):
    return "ViT lower" if s["ci_high"] < 0 else ("CNN lower" if s["ci_low"] > 0 else "no significant difference")


def mcnemar_units(D, methods):
    """Paired decisions per unit for McNemar's test: source clip (majority over the frames of the given methods;
    > 50% accepted = accepted, a tie = rejected) or, for diffusion, single images (each image is its own cluster).
    Returns (b, c, units): b = units the ViT rejects but the CNN accepts; c = units the ViT accepts but the CNN
    rejects."""
    import numpy as np
    import pandas as pd
    vit, cnn = D.run(VIT), D.run(CNN)
    tv, tc = D.threshold(vit), D.threshold(cnn)
    parts = []
    for m in methods:
        p = D.pos[m]
        parts.append(pd.DataFrame({"unit": D.test["cluster"].to_numpy()[p],
                                   "vit": vit["logit"][p, 0] <= tv, "cnn": cnn["logit"][p, 0] <= tc}))
    f = pd.concat(parts)
    share = f.groupby("unit")[["vit", "cnn"]].mean()
    acc_v, acc_c = (share["vit"] > 0.5).to_numpy(), (share["cnn"] > 0.5).to_numpy()
    return int(np.sum(~acc_v & acc_c)), int(np.sum(acc_v & ~acc_c)), int(len(share))


def h1(D, bt, sb, log):
    import numpy as np
    from ekyc.bootstrap import ci, holm, mcnemar
    from ekyc.metrics import aar, auc, bpcer, threshold_at_bpcer
    vit, cnn = D.run(VIT), D.run(CNN)
    tv, tc = D.threshold(vit), D.threshold(cnn)
    point_aar = {n: {m: aar(D.logits(r, m), t) for m in DF40} for n, r, t in ((VIT, vit, tv), (CNN, cnn, tc))}
    real_v, real_c = D.logits(vit, "cdf_real"), D.logits(cnn, "cdf_real")
    to_v, to_c = threshold_at_bpcer(real_v), threshold_at_bpcer(real_c)  # oracle (Celeb-DF reals)
    oracle_point = {VIT: {m: aar(D.logits(vit, m), to_v) for m in DF40}, CNN: {m: aar(D.logits(cnn, m), to_c)
                                                                                for m in DF40}}
    d_bpcer = summarise(bt.rate("cdf_real", vit_name(D)) - bt.rate("cdf_real", cnn_name(D)),
                        bpcer(real_v, tv) - bpcer(real_c, tc))
    rows, method_rows = [], []
    for kind, (m1, m2) in TYPES.items():
        d = bt.type_aar(vit_name(D), kind) - bt.type_aar(cnn_name(D), kind)
        s = summarise(d, type_value(point_aar[VIT], kind) - type_value(point_aar[CNN], kind))
        od = type_value({m: sb.oracle[(vit_name(D), m)] for m in TYPES[kind]}, kind) \
            - type_value({m: sb.oracle[(cnn_name(D), m)] for m in TYPES[kind]}, kind)
        so = summarise(od, type_value(oracle_point[VIT], kind) - type_value(oracle_point[CNN], kind))
        sens = type_value({m: sb.sens[(vit_name(D), m)] for m in TYPES[kind]}, kind) \
            - type_value({m: sb.sens[(cnn_name(D), m)] for m in TYPES[kind]}, kind)
        s_sens = ci(sens) if np.isfinite(sens).any() else (float("nan"), float("nan"))
        per_method = {}
        for m in (m1, m2):
            dm = summarise(bt.rate(m, vit_name(D)) - bt.rate(m, cnn_name(D)), point_aar[VIT][m] - point_aar[CNN][m])
            per_method[m] = dm
            b_m, c_m, u_m = mcnemar_units(D, (m,))
            method_rows.append({"type": kind, "method": m, "aar_vit": point_aar[VIT][m], "aar_cnn": point_aar[CNN][m],
                                "d_aar": dm["estimate"], "ci_low": dm["ci_low"], "ci_high": dm["ci_high"],
                                "p_boot": dm["p"], "verdict": verdict(dm), "mcnemar_b": b_m, "mcnemar_c": c_m,
                                "mcnemar_units": u_m, "p_mcnemar": mcnemar(b_m, c_m),
                                "auc_vit": D.auc_vs_real(vit, m), "auc_cnn": D.auc_vs_real(cnn, m),
                                "oracle_aar_vit": oracle_point[VIT][m], "oracle_aar_cnn": oracle_point[CNN][m]})
        v = verdict(s)
        signs = [np.sign(per_method[m]["estimate"]) for m in (m1, m2)]
        if signs[0] * signs[1] < 0 and all(verdict(per_method[m]) != "no significant difference" for m in (m1, m2)):
            v = "method-dependent"
        b, c, units = mcnemar_units(D, TYPES[kind])
        auc_v = type_value({m: D.auc_vs_real(vit, m) for m in TYPES[kind]}, kind)
        auc_c = type_value({m: D.auc_vs_real(cnn, m) for m in TYPES[kind]}, kind)
        if v == "ViT lower":
            reading = ("ViT lets fewer through at no extra customer cost" if d_bpcer["ci_low"] <= 0 else
                       ("separates better but rejects more genuine faces (trade-off)" if so["ci_high"] < 0 else
                        "calibration shift, not better detection"))
            row = 1 if d_bpcer["ci_low"] <= 0 else (2 if so["ci_high"] < 0 else 3)
        else:
            reading, row = ("no significant difference", 4) if v == "no significant difference" else (v, 5)
        rows.append({"type": kind, "aar_vit": type_value(point_aar[VIT], kind),
                     "aar_cnn": type_value(point_aar[CNN], kind), "d_aar": s["estimate"], "ci_low": s["ci_low"],
                     "ci_high": s["ci_high"], "p_boot": s["p"], "verdict": v,
                     "mcnemar_b": b, "mcnemar_c": c, "mcnemar_units": units, "p_mcnemar": mcnemar(b, c),
                     "auc_vit": auc_v, "auc_cnn": auc_c,
                     "oracle_d_aar": so["estimate"], "oracle_ci_low": so["ci_low"], "oracle_ci_high": so["ci_high"],
                     "sens_ci_low": s_sens[0], "sens_ci_high": s_sens[1],
                     "sens_verdict": ("ViT lower" if s_sens[1] < 0 else "CNN lower" if s_sens[0] > 0
                                      else "no significant difference") if np.isfinite(s_sens[0]) else "n/a",
                     "reading_row": row, "reading": reading})
    joint = all(r["reading_row"] in (1, 2, 3) for r in rows)
    qualified = joint and any(r["reading_row"] in (2, 3) for r in rows)
    ph, pm = holm([r["p_boot"] for r in rows]), holm([r["p_mcnemar"] for r in rows])
    for r, a, b in zip(rows, ph, pm):
        r["p_boot_holm"], r["p_mcnemar_holm"] = (None, None) if joint else (a, b)
        r["mcnemar_agrees"] = (r["p_mcnemar"] < 0.05) == (r["verdict"] in ("ViT lower", "CNN lower"))
    out = {"joint_h1": "supported" + (" (qualified)" if qualified else "") if joint else "rejected",
           "threshold_vit": tv, "threshold_cnn": tc, "d_bpcer_cdf": d_bpcer,
           "bpcer_cdf_vit": bpcer(real_v, tv), "bpcer_cdf_cnn": bpcer(real_c, tc),
           "threshold_robust": all(r["sens_verdict"] in (r["verdict"], "n/a") or
                                   (r["verdict"] == "method-dependent") for r in rows)}
    log.info(f"H1: {out['joint_h1']}; " + "; ".join(f"{r['type']} {r['verdict']} (dAAR {100 * r['d_aar']:+.1f} pp)"
                                                     for r in rows))
    return out, rows, method_rows


def vit_name(D):
    return f"{VIT}_s0"


def cnn_name(D):
    return f"{CNN}_s0"


def h2(D, bt, sb, log):
    import numpy as np
    from ekyc.bootstrap import ci, p_one_sided_below_zero
    from ekyc.metrics import aar, bpcer
    v, c = vit_name(D), cnn_name(D)
    q = range(1, len(LEVELS))

    def unseen_point(name, li):
        r, t = D.runs[name], D.threshold(D.runs[name])
        per = {m: aar(D.logits(r, m, li), t) for m in DF40}
        return sum(type_value(per, k) for k in TYPES) / 3

    def did(f):
        return np.mean([(f(v, li) - f(v, 0)) - (f(c, li) - f(c, 0)) for li in q], axis=0)

    d = did(bt.unseen_aar)
    point = did(unseen_point)
    lo, hi = ci(d)
    out = {"did_aar": float(point), "ci_low": lo, "ci_high": hi, "p_one_sided": p_one_sided_below_zero(d),
           "verdict": "supported" if hi < 0 else "rejected"}
    q30 = LEVELS.index("q30")
    out["d_aar_q30"] = summarise(bt.unseen_aar(v, q30) - bt.unseen_aar(c, q30), unseen_point(v, q30) - unseen_point(c, q30))
    bp = did(lambda n, li: bt.rate("cdf_real", n, li))
    bp_point = did(lambda n, li: bpcer(D.logits(D.runs[n], "cdf_real", li), D.threshold(D.runs[n])))
    out["did_bpcer_cdf"] = summarise(bp, bp_point)
    au = did(lambda n, li: sb.unseen_auc(n, li))
    out["did_auc"] = summarise(au, did(lambda n, li: D.unseen_auc(D.runs[n], li)))
    out["robust_reading"] = (out["verdict"] == "supported" and out["did_bpcer_cdf"]["ci_low"] <= 0
                             and out["did_auc"]["estimate"] >= 0)
    out["degrades_more_but_stays_better"] = out["verdict"] == "rejected" and out["d_aar_q30"]["ci_high"] < 0
    out["curves"] = {n: [unseen_point(n, li) for li in range(len(LEVELS))] for n in (v, c)}
    retention = {}
    for n in (v, c):
        r, t = D.runs[n], D.threshold(D.runs[n])
        caught_png = np.concatenate([D.logits(r, m, 0) > t for m in DF40])
        caught_q30 = np.concatenate([D.logits(r, m, q30) > t for m in DF40])
        retention[n] = float(caught_q30[caught_png].mean()) if caught_png.any() else float("nan")
    out["retention_png_to_q30"] = retention
    log.info(f"H2: {out['verdict']} (DiD {100 * point:+.2f} pp, CI {100 * lo:+.2f} to {100 * hi:+.2f})")
    return out


def rq3(D, bt, sb):
    from ekyc.metrics import aar
    rows = []
    v = vit_name(D)
    for other in (f"{PROBE}_s0", ZS):
        if other not in D.runs:
            continue
        for kind in TYPES:
            pts = {}
            for n in (v, other):
                r, t = D.runs[n], D.threshold(D.runs[n])
                pts[n] = type_value({m: aar(D.logits(r, m), t) for m in DF40}, kind)
            s = summarise(bt.type_aar(v, kind) - bt.type_aar(other, kind), pts[v] - pts[other])
            da = sb.type_auc(v, kind) - sb.type_auc(other, kind)
            sa = summarise(da, type_value({m: D.auc_vs_real(D.runs[v], m) - D.auc_vs_real(D.runs[other], m)
                                           for m in TYPES[kind]}, kind))
            rows.append({"comparison": f"ViT - {NAMES[D.runs[other]['kind']]}", "type": kind,
                         "aar_vit": pts[v], "aar_other": pts[other], "d_aar": s["estimate"], "ci_low": s["ci_low"],
                         "ci_high": s["ci_high"], "d_auc": sa["estimate"], "d_auc_ci_low": sa["ci_low"],
                         "d_auc_ci_high": sa["ci_high"]})
    return rows


def fairness(D, B, log):
    """BPCER on Celeb-DF test real frames by apparent gender, identity-cluster bootstrap (pre-registered primary);
    plus exploratory AAR by gender."""
    import numpy as np
    from ekyc.bootstrap import ci, cluster_sums, rate_reps, weights
    from ekyc.metrics import aar, bpcer
    rows, explore = [], []
    real = D.pos["cdf_real"]
    ids = D.test["identity"].to_numpy()[real]
    gender = D.test["gender"].astype(str).to_numpy()[real]
    names = sorted(set(ids))
    code = np.array([names.index(i) for i in ids])
    for kind in (VIT, CNN, PROBE, ZS):
        name = ZS if kind == ZS else f"{kind}_s0"
        if name not in D.runs:
            continue
        r, t = D.runs[name], D.threshold(D.runs[name])
        rej = r["logit"][real, 0] > t
        sums = {g: cluster_sums(code[gender == g], len(names), rej[gender == g]) for g in ("F", "M")}
        reps = {g: [] for g in ("F", "M")}
        for W in weights("ID_CDF", len(names), B):
            for g in ("F", "M"):
                reps[g].append(rate_reps(W, *sums[g])[:, 0])
        rf, rm = np.concatenate(reps["F"]), np.concatenate(reps["M"])
        pf, pm = bpcer(r["logit"][real, 0][gender == "F"], t), bpcer(r["logit"][real, 0][gender == "M"], t)
        lo, hi = ci(rf - rm)
        with np.errstate(divide="ignore", invalid="ignore"):
            rlo, rhi = ci(np.where(rm > 0, rf / rm, np.nan))
        videos = D.test["video"].to_numpy()[real]
        rows.append({"model": NAMES[kind], "run": name, "bpcer_female": pf, "bpcer_male": pm, "gap_f_minus_m": pf - pm,
                     "ci_low": lo, "ci_high": hi, "ratio_f_over_m": pf / pm if pm else float("nan"),
                     "ratio_ci_low": rlo, "ratio_ci_high": rhi, "ci_half_width": (hi - lo) / 2,
                     "identities_f": len(set(ids[gender == "F"])), "identities_m": len(set(ids[gender == "M"])),
                     "videos_f": len(set(videos[gender == "F"])), "videos_m": len(set(videos[gender == "M"])),
                     "frames_f": int((gender == "F").sum()), "frames_m": int((gender == "M").sum())})
        for g_name in ("cdf_fake", "inswap", "simswap", "fomm", "sadtalker", "ffpp_real"):
            p = D.pos[g_name]
            gg = D.test["gender"].astype(str).to_numpy()[p]
            x = r["logit"][p, 0]
            metric = "BPCER" if g_name.endswith("_real") else "AAR"
            f = (lambda z: bpcer(z, t)) if metric == "BPCER" else (lambda z: aar(z, t))
            for g in ("F", "M"):
                if (gg == g).any():
                    explore.append({"model": NAMES[kind], "group": g_name, "metric": metric, "gender": g,
                                    "value": f(x[gg == g]), "frames": int((gg == g).sum()),
                                    "videos": int(len(set(D.test["video"].to_numpy()[p][gg == g])))})
    log.info("Fairness (Celeb-DF real BPCER, F - M): " + "; ".join(
        f"{r['model']} {100 * r['gap_f_minus_m']:+.1f} pp [{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}]"
        for r in rows))
    return rows, explore


def tradeoff(D):
    import numpy as np
    from ekyc.metrics import aar, bpcer
    rows = []
    for kind in (VIT, CNN, PROBE, ZS):
        name = ZS if kind == ZS else f"{kind}_s0"
        if name not in D.runs:
            continue
        r = D.runs[name]
        ts = {target: D.threshold(r, target) for target in TRADEOFF}
        for target, t in ts.items():
            per = {m: aar(D.logits(r, m), t) for m in DF40}
            rows.append({"model": NAMES[kind], "target_val_bpcer": target, "threshold": t,
                         **{f"aar_{k}": type_value(per, k) for k in TYPES},
                         "aar_unseen": sum(type_value(per, k) for k in TYPES) / 3,
                         "bpcer_cdf": bpcer(D.logits(r, "cdf_real"), t),
                         "bpcer_ffpp_test": bpcer(D.logits(r, "ffpp_real"), t)})
        lo, hi = ts[0.05], ts[0.005]  # frames between the 5% and 0.5% thresholds: a human-review band
        band_real = D.logits(r, "cdf_real")
        band_fake = np.concatenate([D.logits(r, m) for m in DF40])
        rows[-len(ts)]["review_band_cdf_real"] = float(((band_real > lo) & (band_real <= hi)).mean())
        rows[-len(ts)]["review_band_df40_fake"] = float(((band_fake > lo) & (band_fake <= hi)).mean())
    return rows


def seeds_table(D):
    import numpy as np
    from ekyc.metrics import aar, auc, bpcer
    rows = []
    for seed in D.seeds:
        names = {k: f"{k}_s{seed}" for k in (VIT, CNN, PROBE)}
        if not all(n in D.runs for n in names.values()):
            continue
        vals = {}
        for k, n in names.items():
            r, t = D.runs[n], D.threshold(D.runs[n])
            per = {m: aar(D.logits(r, m), t) for m in DF40}
            real = D.logits(r, "cdf_real")
            fake = D.logits(r, "cdf_fake")
            vals[k] = {**{f"aar_{kk}": type_value(per, kk) for kk in TYPES},
                       "aar_unseen": sum(type_value(per, kk) for kk in TYPES) / 3,
                       "bpcer_cdf": bpcer(real, t), "aar_cdf": aar(fake, t),
                       "auc_cdf": auc(np.r_[real, fake], np.r_[np.zeros(len(real)), np.ones(len(fake))]),
                       "threshold": t, "val_auc": r["info"].get("val_auc")}
            rows.append({"seed": seed, "model": NAMES[k], **vals[k]})
        rows.append({"seed": seed, "model": "ViT - CNN", **{f"aar_{kk}": vals[VIT][f"aar_{kk}"] - vals[CNN][f"aar_{kk}"]
                                                            for kk in TYPES},
                     "aar_unseen": vals[VIT]["aar_unseen"] - vals[CNN]["aar_unseen"]})
    return rows


def to_md_table(df, pct=(), digits=3):
    import math
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            x = r[c]
            if isinstance(x, float) and math.isnan(x):
                cells.append("-")
            elif isinstance(x, float) and c.startswith("p_"):
                cells.append("<0.001" if x < 0.001 else f"{x:.3f}")
            elif isinstance(x, float):
                cells.append(f"{100 * x:.1f}%" if c in pct else f"{x:.{digits}f}")
            else:
                cells.append(str(x))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compute every reported number from the stored scores (step 5).")
    ap.add_argument("--scores", type=Path, default=OUT / "scores")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--B", type=int, default=10_000, help="bootstrap replicates for rates (AUC etc.: B/5)")
    args = ap.parse_args()
    setup_console()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(OUT / "logs" / "analyze.log")
    import pandas as pd
    t0 = time.time()

    D = Data(args.scores, log)
    thr, frame, conf = point_tables(D)
    for name, df in (("thresholds", thr), ("metrics_frame", frame), ("confusion", conf)):
        atomic_write(out / f"{name}.csv", df.to_csv(index=False, float_format="%.6g", lineterminator="\n"))

    primary = [n for n in (f"{VIT}_s0", f"{CNN}_s0", f"{PROBE}_s0", ZS) if n in D.runs]
    bt = Boot(D, primary, args.B, log)
    b_slow = max(250, args.B // 5)
    auc_cells = [(n, li, m) for n in (f"{VIT}_s0", f"{CNN}_s0") for li in range(len(LEVELS)) for m in DF40]
    auc_cells += [(n, 0, m) for n in primary if n not in (f"{VIT}_s0", f"{CNN}_s0") for m in DF40]
    sb = SlowBoot(D, auc_cells, [f"{VIT}_s0", f"{CNN}_s0"], [f"{VIT}_s0", f"{CNN}_s0"], b_slow, log)

    h1_out, h1_rows, h1_methods = h1(D, bt, sb, log)
    h2_out = h2(D, bt, sb, log)
    rq3_rows = rq3(D, bt, sb)
    fair_rows, fair_explore = fairness(D, args.B, log)
    trade_rows = tradeoff(D)
    seed_rows = seeds_table(D)
    sanity = []
    for seed in D.seeds:
        n = f"{CNN}_s{seed}"
        if n in D.runs:
            a = float(frame.query("run == @n and level == 'png' and group == 'cdf_fake' and metric == 'AUC'")["value"].iloc[0])
            sanity.append({"seed": seed, "effnet_cdf_frame_auc": a, "band": sanity_band(a), "reference": SANITY_REF})
    from ekyc.bootstrap import ci
    ff_test_bpcer = {}
    for n in primary:
        x = D.logits(D.runs[n], "ffpp_real")
        ff_test_bpcer[n] = {"bpcer": float((x > D.threshold(D.runs[n])).mean()),
                            "ci": ci(bt.rate("ffpp_real", n))}
    rate_rows = []  # every rate with its CI, per primary run and level (for the figures)
    for n in primary:
        r, t = D.runs[n], D.threshold(D.runs[n])
        for li, level in enumerate(LEVELS):
            for g in GROUPS:
                x = D.logits(r, g, li)
                real = g.endswith("_real")
                lo, hi = ci(bt.rate(g, n, li))
                rate_rows.append({"run": n, "kind": r["kind"], "level": level, "group": g,
                                  "metric": "BPCER" if real else "AAR",
                                  "value": float((x > t).mean() if real else (x <= t).mean()),
                                  "ci_low": lo, "ci_high": hi})
            per = {m: float((D.logits(r, m, li) <= t).mean()) for m in DF40}
            for k in TYPES:
                lo, hi = ci(bt.type_aar(n, k, li))
                rate_rows.append({"run": n, "kind": r["kind"], "level": level, "group": f"type:{k}", "metric": "AAR",
                                  "value": type_value(per, k), "ci_low": lo, "ci_high": hi})
            lo, hi = ci(bt.unseen_aar(n, li))
            rate_rows.append({"run": n, "kind": r["kind"], "level": level, "group": "unseen", "metric": "AAR",
                              "value": sum(type_value(per, k) for k in TYPES) / 3, "ci_low": lo, "ci_high": hi})

    tables = {"h1": pd.DataFrame(h1_rows), "h1_methods": pd.DataFrame(h1_methods), "rq3": pd.DataFrame(rq3_rows),
              "fairness": pd.DataFrame(fair_rows), "fairness_exploratory": pd.DataFrame(fair_explore),
              "tradeoff": pd.DataFrame(trade_rows), "seeds": pd.DataFrame(seed_rows), "sanity": pd.DataFrame(sanity),
              "rates_ci": pd.DataFrame(rate_rows)}
    for name, df in tables.items():
        atomic_write(out / f"{name}.csv", df.to_csv(index=False, float_format="%.6g", lineterminator="\n"))
    summary = {"computed": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(), "B": args.B,
               "B_slow": b_slow, "limited_test_run": D.limited, "runs": list(D.runs), "h1": h1_out, "h2": h2_out,
               "ffpp_test_bpcer": ff_test_bpcer, "sanity": sanity,
               "clusters": {s: len(c) for s, c in D.strata.items()} | {"VAL": len(D.val_names)}}
    atomic_write(out / "summary.json", json.dumps(summary, indent=1, default=float))

    pct = ("aar_vit", "aar_cnn", "d_aar", "ci_low", "ci_high", "oracle_d_aar", "oracle_ci_low", "oracle_ci_high",
           "aar_other", "bpcer_female", "bpcer_male", "gap_f_minus_m", "aar_faceswap", "aar_reenactment",
           "aar_diffusion", "aar_unseen", "bpcer_cdf", "bpcer_ffpp_test", "aar_cdf", "sens_ci_low", "sens_ci_high",
           "review_band_cdf_real", "review_band_df40_fake")
    md = [f"# Results (computed {summary['computed']}, commit {summary['git_commit']}, B = {args.B:,})", "",
          "Generated by analyze.py from outputs/scores; Percentages are of frames.", "",
          f"**H1 (joint): {h1_out['joint_h1']}.** Celeb-DF real BPCER at the fixed thresholds: ViT "
          f"{100 * h1_out['bpcer_cdf_vit']:.1f}%, CNN {100 * h1_out['bpcer_cdf_cnn']:.1f}%.", "",
          to_md_table(tables["h1"][["type", "aar_vit", "aar_cnn", "d_aar", "ci_low", "ci_high", "verdict",
                                    "p_mcnemar", "auc_vit", "auc_cnn", "oracle_d_aar", "reading"]], pct), "",
          f"**H2: {h2_out['verdict']}.** DiD {100 * h2_out['did_aar']:+.2f} pp (CI {100 * h2_out['ci_low']:+.2f} to "
          f"{100 * h2_out['ci_high']:+.2f}).", "", "## RQ3", "", to_md_table(tables["rq3"], pct) if rq3_rows else "-",
          "", "## Fairness (Celeb-DF test real frames, female - male BPCER)", "",
          to_md_table(tables["fairness"][["model", "bpcer_female", "bpcer_male", "gap_f_minus_m", "ci_low", "ci_high",
                                          "identities_f", "identities_m"]], pct) if fair_rows else "-",
          "", "## Trade-off", "", to_md_table(tables["tradeoff"], pct), "", "## Seeds", "",
          to_md_table(tables["seeds"], pct) if seed_rows else "-", "", "## Sanity check", "",
          to_md_table(tables["sanity"]) if sanity else "-", ""]
    atomic_write(out / "RESULTS.md", "\n".join(md))
    log.info(f"[ OK ] analysis written to {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
