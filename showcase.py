"""
showcase.py - face visuals for the report and the pitch. PRIVATE: they show dataset faces (FF++, Celeb-DF and DF40
are research-only, no redistribution), so they are saved to outputs/samples/ and never go on GitHub.

Every face is labelled genuine or deepfake, and each model's decision comes from the stored scores at its fixed 1%
threshold (seed 0, thresholds.csv from analyze.py). Faces are drawn with a fixed seed from the sharper half of a
random sample, so the examples are reproducible and not hand-picked.

  showcase_spot_the_fake_quiz.png    one genuine face and one deepfake per attack type, lettered A-D, no answers
  showcase_spot_the_fake_answer.png  the same, labelled, with what each model decided
  showcase_compression.png           two fakes at original quality and JPEG 90/60/30, with each model's decision
  showcase_heatmaps.png              per attack type, a fake the ViT caught and one it missed, with the ViT's attention
                                     rollout and EfficientNet's Grad-CAM map
  showcase_rejected_genuine.png      genuine Celeb-DF faces that the fixed thresholds wrongly reject
  showcase_manifest.json             every face shown: index row, file, label, method, scores and decisions

Usage (VS Code terminal, inside the code folder; after analyze.py):
    .venv\\Scripts\\python.exe -B showcase.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import DATA, LEVELS, OUT, RESULTS, SAMPLES  # noqa: E402
from ekyc.utils import atomic_write, setup_console  # noqa: E402

SEED = 11
MARGIN = 0.5  # examples must be decided clearly: at least this many logits from each model's threshold
NAMES = {"clip_lora": "ViT", "effnet_b4": "EfficientNet"}
METHOD = {"inswap": "InSwapper", "simswap": "SimSwap", "fomm": "FOMM", "sadtalker": "SadTalker", "sd2.1": "SD-2.1",
          "DiT": "DiT-XL/2"}
TYPE_OF = {"inswap": "face swap", "simswap": "face swap", "fomm": "reenactment", "sadtalker": "reenactment",
           "sd2.1": "diffusion", "DiT": "diffusion"}
TYPES = {"face swap": ("inswap", "simswap"), "reenactment": ("fomm", "sadtalker"), "diffusion": ("sd2.1", "DiT")}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GOOD, BAD = "#0ca30c", "#d03b3b"  # status colours: always with a tick/cross and a word, never colour alone
CREDIT = ("Faces: Celeb-DF v2 (Li et al., 2020) and DF40 (Yan et al., 2024; CC BY-NC 4.0), research datasets, resized; "
          "shown for this assessment only, not for redistribution. Decisions: each model's fixed 1% threshold (seed 0).")


class Scores:
    def __init__(self):
        import numpy as np
        import pandas as pd
        thr = pd.read_csv(RESULTS / "thresholds.csv").query("target_bpcer == 0.01").set_index("run")["threshold"]
        self.t, self.logit = {}, {}
        for kind in NAMES:
            z = np.load(OUT / "scores" / f"{kind}_s0" / "scores.npz")
            self.logit[kind] = pd.DataFrame(z["logit"], index=z["row"], columns=list(LEVELS))
            self.t[kind] = float(thr[f"{kind}_s0"])

    def fake(self, kind, rows, level="png"):
        """True where the model calls the frame fake (logit above its fixed threshold)."""
        return self.logit[kind].loc[rows, level].to_numpy() > self.t[kind]

    def record(self, row, levels=("png",)):
        return {NAMES[k]: {lv: {"logit": round(float(self.logit[k].at[row, lv]), 4),
                                "called": "fake" if self.logit[k].at[row, lv] > self.t[k] else "genuine"}
                           for lv in levels} for k in NAMES}


def image(idx, row, level="png"):
    from ekyc.data import decode_256, jpeg
    img = decode_256(str(DATA / idx.at[row, "path"]), idx.at[row, "src_res"] == 512)
    q = None if level == "png" else int(level[1:])
    return img if q is None else jpeg(img, q)


def black_border(img, width=12, dark=12, share=0.01, corner=16):
    """True if the crop has a black band, or a black wedge in a corner (alignment fill), which would give a fake away
    on a slide."""
    lum = img.numpy().max(0)
    frame = np.ones(lum.shape, dtype=bool)
    frame[width:-width, width:-width] = False
    corners = [lum[:corner, :corner], lum[:corner, -corner:], lum[-corner:, :corner], lum[-corner:, -corner:]]
    return float((lum[frame] < dark).mean()) > share or any(float((c < dark).mean()) > 0.2 for c in corners)


def people(idx, row):
    """The Celeb-DF people behind a frame: the id(s) in its folder name (both people of a swap), the YouTube-real
    video, or none for the diffusion images (unrelated random batches)."""
    if idx.at[row, "method"] in ("sd2.1", "DiT"):
        return set()
    folder = idx.at[row, "path"].split("/")[-2]
    ids = {t for t in folder.split("_") if t[:2] == "id" and t[2:].isdigit()}
    return ids or {f"yt{folder}"}


def pick(idx, rows, rng, n=1, used=None, sample=120):
    """n frames showing people not used yet (in any figure), drawn at random from the sharper half of a random
    sample of rows, skipping crops with a black border. Adds the people picked to `used`."""
    used = set() if used is None else used
    rows = np.asarray([r for r in rows if not people(idx, r) & used])
    if not len(rows):
        raise ValueError("no frames meet the conditions")
    cand = rng.choice(rows, min(sample, len(rows)), replace=False)
    sharp = []
    for r in cand:
        img = image(idx, r)
        y = img.float().mean(0).numpy()
        lap = y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] - 4 * y[1:-1, 1:-1]
        sharp.append(-1.0 if black_border(img) else lap.var())
    cand = cand[np.asarray(sharp) >= 0]
    sharp = [s for s in sharp if s >= 0]
    if not len(cand):
        raise ValueError("every sampled frame has a black border")
    top = cand[np.argsort(sharp)[::-1][:max(n, len(cand) // 2)]]
    out = []
    for r in rng.permutation(top):
        if not people(idx, r) & used:
            out.append(int(r))
            used |= people(idx, r)
        if len(out) == n:
            return out
    raise ValueError("not enough frames of distinct people")


def label_of(idx, row):
    m = idx.at[row, "method"]
    return "Genuine" if idx.at[row, "label"] == 0 else f"Deepfake: {METHOD[m]} ({TYPE_OF[m]})"


def verdicts(ax, S, idx, row, level="png", y0=-0.06, size=9):
    """Two lines under an image: what each model called it, with a tick if right and a cross if wrong."""
    truth_fake = idx.at[row, "label"] == 1
    for i, kind in enumerate(NAMES):
        called_fake = bool(S.fake(kind, [row], level)[0])
        right = called_fake == truth_fake
        ax.text(0.02, y0 - 0.11 * i, "✓" if right else "✗", transform=ax.transAxes, ha="left", va="top",
                fontsize=size + 2, color=GOOD if right else BAD, fontweight="bold")
        ax.text(0.13, y0 - 0.11 * i, f"{NAMES[kind]}: {'fake' if called_fake else 'real'}", transform=ax.transAxes,
                ha="left", va="top", fontsize=size, color=INK)


def show(ax, img):
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def credit(fig, y=0.01, decisions=True):
    text = CREDIT if decisions else CREDIT.split(" Decisions:")[0]
    fig.text(0.01, y, text, fontsize=6.5, color=MUTED, ha="left", va="bottom")


def main():
    setup_console()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from ekyc.data import load_index
    from ekyc.models import load_checkpoint, set_eager_attention
    from explain_maps import BLUES, maps_for, stretch

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    idx = load_index(columns=["path", "set", "method", "label", "identity", "gender", "src_res"])
    S = Scores()
    rng = np.random.default_rng(SEED)
    scored = S.logit["clip_lora"].index
    sub = idx.loc[scored]
    manifest = {"seed": SEED, "thresholds": {NAMES[k]: S.t[k] for k in NAMES}, "figures": {}}

    def rows_where(method=None, label=None, set_=None):
        m = np.ones(len(sub), dtype=bool)
        if method is not None:
            m &= (sub["method"] == method).to_numpy()
        if label is not None:
            m &= (sub["label"] == label).to_numpy()
        if set_ is not None:
            m &= (sub["set"] == set_).to_numpy()
        return sub.index.to_numpy()[m]

    def clear(kind, rows, fake, level="png"):
        """Rows a model decides clearly (more than MARGIN logits from its threshold), as fake or as genuine."""
        x = S.logit[kind].loc[rows, level].to_numpy() - S.t[kind]
        return x > MARGIN if fake else x < -MARGIN

    def both(rows, fake, level="png"):
        keep = np.ones(len(rows), dtype=bool)
        for k in NAMES:
            keep &= clear(k, rows, fake, level)
        return rows[keep]

    used = set()  # people already shown in any figure

    # 1. Spot the fake: a genuine face both models accept, and a fake per attack type that fooled both models
    real = both(rows_where("real", 0, "cdf_test"), False)
    chosen = pick(idx, real, rng, used=used)
    for m in ("simswap", "sadtalker", "DiT"):
        chosen += pick(idx, both(rows_where(m), False), rng, used=used)
    order = [chosen[i] for i in rng.permutation(len(chosen))]
    for answer in (False, True):
        fig, axes = plt.subplots(1, 4, figsize=(12, 4.6 if answer else 3.9))
        for i, (ax, row) in enumerate(zip(axes, order)):
            show(ax, image(idx, row))
            ax.set_title("ABCD"[i], fontsize=16, color=INK, fontweight="bold", pad=6)
            if answer:
                ax.text(0.5, -0.04, label_of(idx, row), transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                        color=INK, fontweight="bold")
                verdicts(ax, S, idx, row, y0=-0.15)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.27 if answer else 0.08, wspace=0.06)
        credit(fig, decisions=answer)
        fig.savefig(SAMPLES / f"showcase_spot_the_fake_{'answer' if answer else 'quiz'}.png", dpi=200,
                    facecolor="white")
        plt.close(fig)
    manifest["figures"]["spot_the_fake"] = [{"row": r, "path": idx.at[r, "path"], "shown_as": label_of(idx, r),
                                             "letter": "ABCD"[i], **S.record(r)} for i, r in enumerate(order)]

    # 2. Compression: fakes both models catch at full quality but accept once the image is JPEG-compressed
    levels = ("png", "q90", "q60", "q30")
    comp = []
    for methods in (("inswap", "simswap"), ("sd2.1",)):
        rows = np.concatenate([rows_where(m) for m in methods])
        rows = both(both(both(rows, True, "png"), False, "q60"), False, "q30")
        rows = rows[np.all([clear(k, rows, True, "q90") | clear(k, rows, False, "q90") for k in NAMES], axis=0)]
        comp += pick(idx, rows, rng, used=used)
    fig, axes = plt.subplots(len(comp), 4, figsize=(9.6, 5.9))
    for r_i, row in enumerate(comp):
        for c, lv in enumerate(levels):
            ax = axes[r_i, c]
            show(ax, image(idx, row, lv))
            if r_i == 0:
                ax.set_title("Original" if lv == "png" else f"JPEG quality {lv[1:]}", fontsize=10, color=INK)
            verdicts(ax, S, idx, row, lv, y0=-0.03, size=8.5)
        axes[r_i, 0].text(-0.08, 0.5, label_of(idx, row).replace(": ", ":\n"), transform=axes[r_i, 0].transAxes,
                          ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.94, bottom=0.09, wspace=0.05, hspace=0.3)
    credit(fig)
    fig.savefig(SAMPLES / "showcase_compression.png", dpi=200, facecolor="white")
    plt.close(fig)
    manifest["figures"]["compression"] = [{"row": r, "path": idx.at[r, "path"], "shown_as": label_of(idx, r),
                                           **S.record(r, levels)} for r in comp]

    # 3. Heat maps: per attack type, a fake the ViT caught and one it missed
    models = {k: load_checkpoint(OUT / "checkpoints" / f"{k}_s0" / "best.safetensors")[0] for k in NAMES}
    set_eager_attention(models["clip_lora"])
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    heat = {}
    for t, methods in TYPES.items():
        rows = np.concatenate([rows_where(m) for m in methods])
        for outcome, sel in (("caught", clear("clip_lora", rows, True)), ("missed", clear("clip_lora", rows, False))):
            heat[(t, outcome)] = pick(idx, rows[sel], rng, used=used)[0]
    fig, axes = plt.subplots(2, 9, figsize=(15, 4.9))
    for c_t, t in enumerate(TYPES):
        for r_o, outcome in enumerate(("caught", "missed")):
            row = heat[(t, outcome)]
            img = image(idx, row)
            gray = img.float().mean(0).numpy() / 255
            maps = {k: maps_for(models[k], k, [idx.at[row, "path"]], [idx.at[row, "src_res"] == 512])[0][0]
                    for k in NAMES}
            panels = [(None, label_of(idx, row).split(" (")[0].replace("Deepfake: ", "")),
                      (maps["clip_lora"], "ViT rollout"), (maps["effnet_b4"], "EfficientNet Grad-CAM")]
            for j, (m, title) in enumerate(panels):
                ax = axes[r_o, 3 * c_t + j]
                if m is None:
                    show(ax, img)
                else:  # the map over a faded grey copy of the face
                    ax.imshow(gray, cmap="gray", vmin=0, vmax=1, alpha=0.45)
                    ax.imshow(stretch(m), cmap=cmap, vmin=0, vmax=1, alpha=0.75)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_visible(False)
                ax.set_title(title, fontsize=8.5, color=INK2 if m is not None else INK,
                             fontweight="normal" if m is not None else "bold")
            verdicts(axes[r_o, 3 * c_t], S, idx, row, y0=-0.03, size=8)
        axes[0, 3 * c_t + 1].text(0.5, 1.32, t.capitalize(), transform=axes[0, 3 * c_t + 1].transAxes, ha="center",
                                  va="bottom", fontsize=11, color=INK, fontweight="bold")
    for r_o, outcome in enumerate(("Caught by the ViT", "Missed by the ViT")):
        axes[r_o, 0].text(-0.12, 0.5, outcome, transform=axes[r_o, 0].transAxes, rotation=90, ha="right",
                          va="center", fontsize=10, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.995, top=0.84, bottom=0.12, wspace=0.05, hspace=0.55)
    credit(fig)
    fig.savefig(SAMPLES / "showcase_heatmaps.png", dpi=200, facecolor="white")
    plt.close(fig)
    manifest["figures"]["heatmaps"] = [{"row": r, "path": idx.at[r, "path"], "shown_as": label_of(idx, r),
                                        "group": f"{t}, {o} by the ViT", **S.record(r)}
                                       for (t, o), r in heat.items()]

    # 4. Genuine faces the fixed thresholds wrongly reject (customers who would be locked out)
    rejected = both(rows_where("real", 0, "cdf_test"), True)
    gen = []
    for g in ("F", "M", "F", "M"):  # apparent-gender balance only; the labels are not shown
        pool = rejected[(idx.loc[rejected, "gender"].astype(str) == g).to_numpy()]
        gen += pick(idx, pool if len(pool) else rejected, rng, used=used)
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["h1"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.5))
    for ax, row in zip(axes, gen):
        show(ax, image(idx, row))
        ax.text(0.5, -0.04, "Genuine (Celeb-DF)", transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                color=INK, fontweight="bold")
        verdicts(ax, S, idx, row, y0=-0.15)
    fig.text(0.5, 0.97, f"Genuine Celeb-DF frames rejected at the thresholds fixed on FF++ (target 1%): "
             f"ViT {100 * summary['bpcer_cdf_vit']:.1f}%, EfficientNet {100 * summary['bpcer_cdf_cnn']:.1f}%",
             ha="center", va="top", fontsize=10, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.27, wspace=0.06)
    credit(fig)
    fig.savefig(SAMPLES / "showcase_rejected_genuine.png", dpi=200, facecolor="white")
    plt.close(fig)
    manifest["figures"]["rejected_genuine"] = [{"row": r, "path": idx.at[r, "path"], "shown_as": "Genuine (Celeb-DF)",
                                                **S.record(r)} for r in gen]

    atomic_write(SAMPLES / "showcase_manifest.json", json.dumps(manifest, indent=1))
    print("[ OK ] 5 face figures + showcase_manifest.json in outputs/samples/ (private: never commit)")


if __name__ == "__main__":
    main()
