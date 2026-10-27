"""
h2_seeds.py - exploratory addition after the results freeze (25 Oct 2026; see outputs/results/FROZEN.md): the
pre-registered H2 statistic (difference-in-differences of the overall unseen AAR, JPEG 90-30 against the original
PNG, ViT minus EfficientNet-B4) and its Celeb-DF BPCER and unseen-AUC companions, repeated for seeds 1 and 2 with the
same code path and thresholds as analyze.h2(). ANALYSIS_PLAN.md section 7 calls a result robust to training randomness
if the H2 DiD has the same sign in all three seeds, so point estimates are enough (no bootstrap here). Seed 0 is
recomputed and must match outputs/results/summary.json exactly.

Writes outputs/results/h2_seeds.csv (numbers only). Nothing frozen is changed.

Usage (VS Code terminal, inside the code folder; after analyze.py):
    .venv\\Scripts\\python.exe -B h2_seeds.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import LEVELS, OUT, RESULTS  # noqa: E402
from ekyc.utils import atomic_write, setup_console, setup_logging  # noqa: E402

SEEDS = (0, 1, 2)


def main():
    setup_console()
    log = setup_logging(OUT / "logs" / "h2_seeds.log")
    import numpy as np
    import pandas as pd
    import analyze as A
    from ekyc.metrics import aar, bpcer

    D = A.Data(OUT / "scores", log)
    q = range(1, len(LEVELS))
    q60 = LEVELS.index("q60")

    def unseen_point(name, li):
        r, t = D.runs[name], D.threshold(D.runs[name])
        per = {m: aar(D.logits(r, m, li), t) for m in A.DF40}
        return sum(A.type_value(per, k) for k in A.TYPES) / 3

    def cdf_bpcer(name, li):
        return bpcer(D.logits(D.runs[name], "cdf_real", li), D.threshold(D.runs[name]))

    def unseen_auc(name, li):
        return D.unseen_auc(D.runs[name], li)

    rows = []
    for seed in SEEDS:
        v, c = f"{A.VIT}_s{seed}", f"{A.CNN}_s{seed}"

        def did(f):
            return float(np.mean([(f(v, li) - f(v, 0)) - (f(c, li) - f(c, 0)) for li in q]))

        row = {"seed": seed, "vit_run": v, "cnn_run": c, "did_aar": did(unseen_point),
               "did_bpcer_cdf": did(cdf_bpcer), "did_auc": did(unseen_auc),
               "aar_unseen_vit_png": unseen_point(v, 0), "aar_unseen_cnn_png": unseen_point(c, 0),
               "aar_unseen_vit_q60": unseen_point(v, q60), "aar_unseen_cnn_q60": unseen_point(c, q60)}
        rows.append(row)
        log.info(f"seed {seed}: AAR DiD {100 * row['did_aar']:+.2f} pp, BPCER DiD {100 * row['did_bpcer_cdf']:+.2f} pp, "
                 f"AUC DiD {row['did_auc']:+.3f}")
    df = pd.DataFrame(rows)

    # Seed 0 must reproduce the frozen H2 numbers
    h2 = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["h2"]
    s0 = df[df["seed"] == 0].iloc[0]
    for col, ref in (("did_aar", h2["did_aar"]), ("did_bpcer_cdf", h2["did_bpcer_cdf"]["estimate"]),
                     ("did_auc", h2["did_auc"]["estimate"])):
        if abs(s0[col] - ref) > 1e-9:
            raise AssertionError(f"seed 0 {col}: {s0[col]} vs frozen {ref}")
    atomic_write(RESULTS / "h2_seeds.csv", df.to_csv(index=False, float_format="%.6g", lineterminator="\n"))
    log.info(f"Seed 0 matches summary.json; wrote {RESULTS / 'h2_seeds.csv'}")


if __name__ == "__main__":
    main()
