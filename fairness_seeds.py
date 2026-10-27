"""
fairness_seeds.py - exploratory addition after the results freeze (24 Oct 2026; see outputs/results/FROZEN.md): the
pre-registered fairness comparison (Celeb-DF test genuine frames rejected, female minus male apparent gender,
identity-cluster bootstrap) repeated for seeds 1 and 2, with the same code path, thresholds and replicates as
analyze.fairness(). Seed 0 is recomputed and must match outputs/results/fairness.csv exactly.

Writes outputs/results/fairness_seeds.csv (numbers only). Nothing frozen is changed.

Usage (VS Code terminal, inside the code folder; after analyze.py):
    .venv\\Scripts\\python.exe -B fairness_seeds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import OUT, RESULTS  # noqa: E402
from ekyc.utils import atomic_write, setup_console, setup_logging  # noqa: E402

B = 10_000  # as analyze.py
SEEDS = (0, 1, 2)


def main():
    setup_console()
    log = setup_logging(OUT / "logs" / "fairness_seeds.log")
    import numpy as np
    import pandas as pd
    import analyze as A
    from ekyc.bootstrap import ci, cluster_sums, rate_reps, weights
    from ekyc.metrics import bpcer

    D = A.Data(OUT / "scores", log)
    real = D.pos["cdf_real"]
    ids = D.test["identity"].to_numpy()[real]
    gender = D.test["gender"].astype(str).to_numpy()[real]
    names = sorted(set(ids))
    code = np.array([names.index(i) for i in ids])
    rows = []
    for kind in (A.VIT, A.CNN, A.PROBE):
        for seed in SEEDS:
            name = f"{kind}_s{seed}"
            r, t = D.runs[name], D.threshold(D.runs[name])
            x = r["logit"][real, 0]
            sums = {g: cluster_sums(code[gender == g], len(names), (x > t)[gender == g]) for g in ("F", "M")}
            reps = {g: [] for g in ("F", "M")}
            for W in weights("ID_CDF", len(names), B):
                for g in ("F", "M"):
                    reps[g].append(rate_reps(W, *sums[g])[:, 0])
            lo, hi = ci(np.concatenate(reps["F"]) - np.concatenate(reps["M"]))
            pf, pm = bpcer(x[gender == "F"], t), bpcer(x[gender == "M"], t)
            rows.append({"model": A.NAMES[kind], "run": name, "seed": seed, "bpcer_all": bpcer(x, t),
                         "bpcer_female": pf, "bpcer_male": pm, "gap_f_minus_m": pf - pm, "ci_low": lo, "ci_high": hi,
                         "identities_f": len(set(ids[gender == "F"])), "identities_m": len(set(ids[gender == "M"]))})
            log.info(f"{name}: F {100 * pf:.1f}%, M {100 * pm:.1f}%, gap {100 * (pf - pm):+.1f} pp "
                     f"[{100 * lo:+.1f}, {100 * hi:+.1f}]")
    df = pd.DataFrame(rows)

    # Seed 0 must reproduce the frozen table
    frozen = pd.read_csv(RESULTS / "fairness.csv").set_index("run")
    for _, row in df[df["seed"] == 0].iterrows():
        f = frozen.loc[row["run"]]
        for col in ("bpcer_female", "bpcer_male", "gap_f_minus_m", "ci_low", "ci_high"):
            if abs(row[col] - f[col]) > 1e-6:
                raise AssertionError(f"{row['run']} {col}: {row[col]} vs frozen {f[col]}")
    atomic_write(RESULTS / "fairness_seeds.csv", df.to_csv(index=False, float_format="%.6g", lineterminator="\n"))
    log.info(f"Seed 0 matches fairness.csv; wrote {RESULTS / 'fairness_seeds.csv'}")


if __name__ == "__main__":
    main()
