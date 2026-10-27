"""
figures.py - step 5 charts for the report and the pitch, from analyze.py's tables and the stored scores. None of
them shows a face, so they go to outputs/figures/ and are listed in its MANIFEST.txt (what check_repo.py allows on
GitHub).

  F0 method.png             the experiment as a flow diagram
  F1 training.png           training and balanced validation loss per epoch, and validation AUC (seed 0)
  F2 roc_scores.png         (a) Celeb-DF ROC on a log false-positive axis with each model's fixed-threshold point;
                            (b) score distributions (FF++ val real, Celeb-DF real, DF40 fakes) with the threshold
  F3 aar_methods.png        AAR per unseen test set with 95% CIs; hollow markers = oracle threshold (H1)
  F4 compression.png        (a) overall unseen AAR and (b) Celeb-DF BPCER vs JPEG quality, with 95% CIs (H2)
  seeds.png                 ViT - CNN difference in AAR per attack type for each seed (repository)
  slides_aar_types.png, slides_compression.png    the two pitch charts, large type

Colours follow the model everywhere (ViT blue, EfficientNet orange, probe aqua, zero-shot yellow), validated for
colour-blind separation on white; the lighter two always carry a direct label.

Usage (VS Code terminal, inside the code folder; after analyze.py):
    .venv\\Scripts\\python.exe -B figures.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import LEVELS, OUT, RESULTS  # noqa: E402
from ekyc.utils import register_figure, setup_console  # noqa: E402

COLORS = {"clip_lora": "#2a78d6", "effnet_b4": "#eb6834", "clip_probe": "#1baf7a", "clip_zeroshot": "#eda100"}
NAMES = {"clip_lora": "ViT (CLIP + LoRA)", "effnet_b4": "EfficientNet-B4", "clip_probe": "Frozen CLIP + probe",
         "clip_zeroshot": "Zero-shot CLIP"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REAL_LIGHT, REAL_DARK = "#c3c2b7", "#52514e"
TESTS = [("cdf_fake", "Celeb-DF v2"), ("inswap", "InSwapper"), ("simswap", "SimSwap"), ("fomm", "FOMM"),
         ("sadtalker", "SadTalker"), ("sd2.1", "SD-2.1"), ("DiT", "DiT-XL/2")]
TYPE_OF = {"cdf_fake": "Celeb-DF", "inswap": "Face swap", "simswap": "Face swap", "fomm": "Reenactment",
           "sadtalker": "Reenactment", "sd2.1": "Diffusion", "DiT": "Diffusion"}
LEVEL_LABELS = ["PNG", "90", "75", "60", "45", "30"]
DF40 = ("inswap", "simswap", "fomm", "sadtalker", "sd2.1", "DiT")


def style(plt, scale=1.0):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8 * scale, "axes.titlesize": 9 * scale,
        "axes.labelsize": 8 * scale, "xtick.labelsize": 7.5 * scale, "ytick.labelsize": 7.5 * scale,
        "legend.fontsize": 7.5 * scale, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})


def save(fig, name, dpi=300):
    path = OUT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    register_figure(path)
    print(f"[ OK ] outputs/figures/{name}")


def end_labels(ax, items, x_text, gap_frac=0.075, size=7.0):
    """Direct labels at line ends, spread apart where they would collide, with hairline leaders back to the line
    (labels are never stacked on top of each other). items: [(x_end, y_end, text)]."""
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * gap_frac
    items = sorted(items, key=lambda t: t[1])
    pos = []
    for _, y, _ in items:
        pos.append(max(y, pos[-1] + gap) if pos else y)
    over = pos[-1] - (y1 - 0.3 * gap) if pos else 0
    if over > 0:
        pos = [q - over for q in pos]
    for (x, y, text), q in zip(items, pos):
        moved = abs(q - y) > 0.15 * gap
        ax.annotate(text, xy=(x, y), xytext=(x_text, q), textcoords="data", va="center", fontsize=size, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=3) if moved else None)


def pct(ax, axis="x"):
    from matplotlib.ticker import PercentFormatter
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))


def f0_method(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.8, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    boxes = [
        (0, 22, 17, 16, "Train: FF++ c23", "real videos and 4 fake\nmethods (Deepfakes,\nFace2Face, FaceSwap,\nNeuralTextures), ~720 each"),
        (20.5, 22, 22, 16, "3 models, trained identically",
         "ViT-B/16 (CLIP) + LoRA\nEfficientNet-B4 (ImageNet)\nfrozen CLIP + linear probe\nsame frames, order, augmentation"),
        (46, 22, 19, 16, "FF++ validation", "best epoch (val AUC)\nthreshold: 1% of genuine\nframes rejected (44 / 4,477)"),
        (68.5, 22, 31.5, 16, "Unseen tests, threshold frozen",
         "Celeb-DF v2 (518 videos)\nDF40: face swap (InSwapper, SimSwap),\nreenactment (FOMM, SadTalker),\ndiffusion (SD-2.1, DiT-XL/2)"),
        (68.5, 4, 31.5, 12, "× 6 image qualities", "original PNG and JPEG 90, 75, 60, 45, 30\n(+ zero-shot CLIP, no training)"),
        (20.5, 4, 44.5, 12, "Measures",
         "AAR (fakes accepted), BPCER (genuine rejected), AUC\ncluster-bootstrap 95% CIs, McNemar's test, apparent-gender gaps"),
    ]
    for x, y, w, h, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#f4f3f0", ec=AXIS,
                                    lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.35)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=MUTED))

    arrow((17, 30), (20.5, 30))
    arrow((42.5, 30), (46, 30))
    arrow((65, 30), (68.5, 30))
    arrow((84.25, 22), (84.25, 16.5))
    arrow((68.5, 10), (65, 10))
    fig.tight_layout(pad=0.2)
    save(fig, "F0_method.png")
    plt.close(fig)


def f1_training(plt, ckpts):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6))
    hist = {}
    for k in ("clip_lora", "effnet_b4", "clip_probe"):
        f = ckpts / f"{k}_s0" / "history.json"
        if f.is_file():
            hist[k] = (json.loads(f.read_text(encoding="utf-8")),
                       json.loads((ckpts / f"{k}_s0" / "result.json").read_text(encoding="utf-8"))["best_epoch"])
    panels = [("clip_lora", axes[0, 0]), ("effnet_b4", axes[0, 1]), ("clip_probe", axes[1, 0])]
    for k, ax in panels:
        if k not in hist:
            ax.axis("off")
            continue
        h, best = hist[k]
        ep = [e["epoch"] + 1 for e in h]
        ax.plot(ep, [e["train_loss"] for e in h], color=COLORS[k], alpha=0.45, label="training")
        ax.plot(ep, [e["val_loss_bal"] for e in h], color=COLORS[k], label="validation (balanced)")
        ax.plot(best + 1, h[best]["val_loss_bal"], "o", ms=6, color=COLORS[k], mec="white", mew=1.5, zorder=3)
        ax.set_title(NAMES[k], loc="left")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (BCE)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left" if k == "clip_probe" else "upper right")  # the probe's curves fill the top
    ax = axes[1, 1]
    for k, (h, best) in hist.items():
        ep = [e["epoch"] + 1 for e in h]
        ax.plot(ep, [e["val_auc"] for e in h], color=COLORS[k])
        ax.plot(best + 1, h[best]["val_auc"], "o", ms=6, color=COLORS[k], mec="white", mew=1.5, zorder=3)
    ax.set_title("Validation AUC (FF++)", loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel("AUC")
    last = max(len(h) for h, _ in hist.values()) if hist else 1
    ax.set_xlim(0.5, last + 0.5 + 0.55 * last)
    ax.set_xticks([t for t in (1, 5, 10, 15, 20) if t <= last])  # the extra width holds the labels, not epochs
    end_labels(ax, [(len(h), h[-1]["val_auc"], NAMES[k]) for k, (h, _) in hist.items()], last + 0.12 * last)
    fig.text(0.01, 0.005, "Dot = the epoch kept by the pre-registered rule.", fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "F1_training.png")
    plt.close(fig)


def load_scores(scores_dir, runs):
    import numpy as np
    out = {}
    for n in runs:
        f = scores_dir / n / "scores.npz"
        if f.is_file():
            z = np.load(f)
            out[n] = {"row": z["row"], "logit": z["logit"], "val_row": z["val_row"], "val_logit": z["val_logit"]}
    return out


def f2_roc_scores(plt, scores, idx, thr):
    import numpy as np
    from sklearn.metrics import roc_curve
    fig = plt.figure(figsize=(6.8, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    for n, s in scores.items():
        k = n.split("_s")[0] if n != "clip_zeroshot" else n
        sets = idx.loc[s["row"], ["set", "method", "label"]].to_numpy()
        cdf = sets[:, 0] == "cdf_test"
        y, x = sets[cdf, 2].astype(int), s["logit"][cdf, 0]
        fpr, tpr, _ = roc_curve(y, x)
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, color=COLORS[k], lw=1.8, label=NAMES[k])
        t = thr[n]
        ax.plot(max((x[y == 0] > t).mean(), 1e-4), (x[y == 1] > t).mean(), "o", ms=7, color=COLORS[k], mec="white",
                mew=1.5, zorder=4)
    ax.axvline(0.01, color=MUTED, lw=0.8)
    ax.text(0.0106, 1.0, "1% genuine rejected", color=MUTED, fontsize=6.8, va="top")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("genuine frames rejected (false-positive rate, log scale)")
    ax.set_ylabel("fakes caught (true-positive rate)")
    ax.set_title("(a) Celeb-DF v2 test: ROC curves; dots = each model's fixed threshold", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, handlelength=1.6, columnspacing=1.2)
    for col, k in enumerate(("clip_lora", "effnet_b4")):
        n = f"{k}_s0"
        if n not in scores:
            continue
        s = scores[n]
        a = fig.add_subplot(gs[1, col])
        sets = idx.loc[s["row"], ["set", "method", "label"]].to_numpy()
        val_real = s["val_logit"][idx.loc[s["val_row"], "label"].to_numpy() == 0]
        cdf_real = s["logit"][(sets[:, 0] == "cdf_test") & (sets[:, 2] == 0), 0]
        df40 = s["logit"][sets[:, 0] == "df40_test", 0]
        lo, hi = np.percentile(np.r_[val_real, cdf_real, df40], [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        a.hist(val_real, bins=bins, density=True, color=REAL_LIGHT, alpha=0.6, label="FF++ val real")
        a.hist(cdf_real, bins=bins, density=True, histtype="step", lw=1.5, color=REAL_DARK, label="Celeb-DF real")
        a.hist(df40, bins=bins, density=True, histtype="step", lw=1.5, color=COLORS[k], label="DF40 fakes")
        a.set_ylim(0, a.get_ylim()[1] * 1.5)
        a.axvline(thr[n], color=INK, lw=1)
        a.text(thr[n], a.get_ylim()[1] * 0.97, " threshold", fontsize=6.8, color=INK, va="top")
        a.set_title(f"({'b' if col == 0 else 'c'}) {NAMES[k]}: scores", loc="left")
        a.set_xlabel("logit (higher = more likely fake)")
        a.set_yticks([])
        a.grid(False)
        a.legend(loc="upper left", fontsize=6.6)
    fig.tight_layout()
    save(fig, "F2_roc_scores.png")
    plt.close(fig)


def f3_aar(plt, rates, h1m):
    import numpy as np
    runs = [n for n in ("clip_lora_s0", "effnet_b4_s0", "clip_probe_s0", "clip_zeroshot") if n in set(rates["run"])]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    y0 = np.arange(len(TESTS))[::-1].astype(float)
    off = np.linspace(-0.27, 0.27, len(runs))
    for j, n in enumerate(runs):
        k = n.split("_s")[0] if n != "clip_zeroshot" else n
        sub = rates.query("run == @n and level == 'png'").set_index("group")
        v = np.array([sub.loc[g, "value"] for g, _ in TESTS])
        lo = np.array([sub.loc[g, "ci_low"] for g, _ in TESTS])
        hi = np.array([sub.loc[g, "ci_high"] for g, _ in TESTS])
        ax.hlines(y0 + off[j], lo, hi, color=COLORS[k], lw=1.6)
        ax.plot(v, y0 + off[j], "o", ms=6, color=COLORS[k], mec="white", mew=1.2, label=NAMES[k], zorder=3)
        if h1m is not None and k in ("clip_lora", "effnet_b4"):
            col = "oracle_aar_vit" if k == "clip_lora" else "oracle_aar_cnn"
            o = h1m.set_index("method")[col]
            ys = [y0[i] + off[j] for i, (g, _) in enumerate(TESTS) if g in o.index]
            ax.plot([o[g] for g, _ in TESTS if g in o.index], ys, "o", ms=6, mfc="white", mec=COLORS[k], mew=1.4,
                    zorder=2)
    ax.set_yticks(y0)
    ax.set_yticklabels([f"{name}  ({TYPE_OF[g]})" if g != "cdf_fake" else name for g, name in TESTS])
    ax.set_xlim(-0.02, 1.02)
    pct(ax)
    ax.set_xlabel("attack acceptance rate (AAR): share of fake frames accepted as genuine")
    ax.grid(axis="y", visible=False)
    ax.set_title("Unseen attacks at the fixed 1% threshold (seed 0, PNG); lines = 95% CI; hollow = oracle threshold",
                 loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(runs), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "F3_aar_methods.png")
    plt.close(fig)


def f4_compression(plt, rates, names=("clip_lora_s0", "effnet_b4_s0", "clip_probe_s0"), out="F4_compression.png",
                   scale=1.0, size=(6.8, 2.9)):
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=size)
    x = np.arange(len(LEVELS))
    for ax, group, title in ((axes[0], "unseen", "(a) Unseen fakes accepted (AAR)"),
                             (axes[1], "cdf_real", "(b) Celeb-DF genuine rejected (BPCER)")):
        ends = []
        for n in names:
            if n not in set(rates["run"]):
                continue
            k = n.split("_s")[0] if n != "clip_zeroshot" else n
            sub = rates.query("run == @n and group == @group").set_index("level").loc[list(LEVELS)]
            ax.fill_between(x, sub["ci_low"], sub["ci_high"], color=COLORS[k], alpha=0.12, lw=0)
            ax.plot(x, sub["value"], color=COLORS[k], marker="o", ms=5 * scale, mec="white", mew=1.2)
            ends.append((x[-1], sub["value"].iloc[-1], NAMES[k].split(" (")[0]))
        ax.set_xticks(x)
        ax.set_xticklabels(LEVEL_LABELS)
        ax.set_xlabel("image quality (JPEG)")
        ax.set_xlim(-0.3, len(LEVELS) + 0.9)
        ax.set_ylim(0, 1 if group == "unseen" else None)
        pct(ax, "y")
        ax.set_title(title, loc="left")
        end_labels(ax, ends, len(LEVELS) - 0.55, size=7 * scale)
    fig.tight_layout()
    save(fig, out)
    plt.close(fig)


def seeds_figure(plt, seeds):
    import numpy as np
    d = seeds[seeds["model"] == "ViT - CNN"]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    types = [("aar_faceswap", "Face swap"), ("aar_reenactment", "Reenactment"), ("aar_diffusion", "Diffusion"),
             ("aar_unseen", "All unseen")]
    for i, (col, label) in enumerate(types):
        ax.plot(d[col], np.full(len(d), i), "o", ms=6, color=COLORS["clip_lora"], mec="white", mew=1.2)
        for _, r in d.iterrows():
            ax.annotate(f"s{int(r['seed'])}", (r[col], i), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=6.5, color=MUTED)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([t for _, t in types])
    ax.invert_yaxis()
    pct(ax)
    ax.set_xlabel("AAR(ViT) - AAR(EfficientNet); below 0 = ViT lets fewer through")
    ax.set_title("Each training seed (each with its own threshold)", loc="left")
    fig.tight_layout()
    save(fig, "seeds.png")
    plt.close(fig)


def slides(plt, rates):
    import numpy as np
    style(plt, 1.6)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    groups = [("type:faceswap", "Face swap"), ("type:reenactment", "Reenactment"), ("type:diffusion", "Diffusion")]
    runs = [n for n in ("clip_lora_s0", "effnet_b4_s0") if n in set(rates["run"])]
    x = np.arange(len(groups))
    w = 0.3
    for j, n in enumerate(runs):
        k = n.split("_s")[0]
        sub = rates.query("run == @n and level == 'png'").set_index("group")
        v = np.array([sub.loc[g, "value"] for g, _ in groups])
        lo = np.array([sub.loc[g, "ci_low"] for g, _ in groups])
        hi = np.array([sub.loc[g, "ci_high"] for g, _ in groups])
        pos = x + (j - 0.5) * (w + 0.04)
        ax.bar(pos, v, width=w, color=COLORS[k], label=NAMES[k])
        ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK2, elinewidth=1.4, capsize=0)
        for p, val in zip(pos, v):
            ax.text(p, val + 0.03, f"{100 * val:.0f}%", ha="center", va="bottom", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([g for _, g in groups])
    ax.set_ylim(0, 1.15)
    pct(ax, "y")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("fakes accepted as genuine")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "slides_aar_types.png", dpi=200)
    plt.close(fig)
    f4_compression(plt, rates, names=("clip_lora_s0", "effnet_b4_s0"), out="slides_compression.png", scale=1.4,
                   size=(11, 4.6))
    style(plt)


def main():
    ap = argparse.ArgumentParser(description="Report and pitch charts (step 5).")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--scores", type=Path, default=OUT / "scores")
    ap.add_argument("--checkpoints", type=Path, default=OUT / "checkpoints")
    args = ap.parse_args()
    setup_console()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from ekyc.data import load_index

    style(plt)
    rates = pd.read_csv(args.results / "rates_ci.csv")
    thr = pd.read_csv(args.results / "thresholds.csv").query("target_bpcer == 0.01").set_index("run")["threshold"]
    h1m = pd.read_csv(args.results / "h1_methods.csv") if (args.results / "h1_methods.csv").is_file() else None
    seeds = pd.read_csv(args.results / "seeds.csv") if (args.results / "seeds.csv").is_file() else pd.DataFrame()
    idx = load_index(columns=["set", "method", "label"])
    scores = load_scores(args.scores, ["clip_lora_s0", "effnet_b4_s0", "clip_probe_s0", "clip_zeroshot"])

    f0_method(plt)
    f1_training(plt, args.checkpoints)
    f2_roc_scores(plt, scores, idx, thr)
    f3_aar(plt, rates, h1m)
    f4_compression(plt, rates)
    if not seeds.empty:
        seeds_figure(plt, seeds)
    slides(plt, rates)


if __name__ == "__main__":
    main()
