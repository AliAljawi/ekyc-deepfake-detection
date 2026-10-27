"""
explain_maps.py - step 5 heat maps: where each model looks when it judges a face (seed 0, original PNG).

- ViT (CLIP + LoRA): attention rollout (Abnar & Zuidema, 2020). Per layer, the attention averaged over heads plus the
  identity (the residual connection), rows renormalised, multiplied through the 12 layers; the CLS token's row over
  the 196 patches (14 x 14) is the map. fp32, eager attention. It shows where information flows to the decision
  token, not why the model says "fake".
- EfficientNet-B4: Grad-CAM (Selvaraju et al., 2017) on the last feature map (8 x 8), for the "fake" logit.
The two kinds of map are not on a common scale; each map is normalised to sum to 1.

Fakes are grouped by attack type and by each model's own decision at its fixed 1% threshold (caught = called fake).

Outputs:
  outputs/samples/explain_grid.png        faces (report and slides only, never GitHub): per attack type, one fake the
                                          ViT caught and one it missed, with both models' maps and decisions, plus a
                                          genuine Celeb-DF face the ViT wrongly rejected
  outputs/figures/explain_mean_maps.png   no faces: each model's mean map over up to 300 caught and 300 missed fakes
                                          per type, with the fixed central face ellipse drawn in
  outputs/results/explain_maps.csv        per model, type and outcome: frames, share of map mass inside the central
                                          ellipse; for the ViT, the share of images with a high-norm patch token
                                          (> 3x the median norm; Darcet et al., 2024)

Usage (VS Code terminal, inside the code folder; after analyze.py):
    .venv\\Scripts\\python.exe -B explain_maps.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import DATA, OUT, RESULTS, SAMPLES  # noqa: E402
from ekyc.utils import atomic_write, register_figure, setup_console  # noqa: E402

TYPES = {"faceswap": ("inswap", "simswap"), "reenactment": ("fomm", "sadtalker"), "diffusion": ("sd2.1", "DiT")}
TYPE_NAMES = {"faceswap": "Face swap", "reenactment": "Reenactment", "diffusion": "Diffusion"}
N_MEAN = 300
SEED = 7
ELLIPSE = (128, 130, 68, 90)  # centre x, centre y, semi-axis x, semi-axis y (pixels of the 256 crop), fixed in advance
INK, INK2, MUTED, HAIRLINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def rollout(model, x_u8):
    """(B, 14, 14) attention rollout of the CLS token, and the share of images with a high-norm patch token."""
    import torch
    from ekyc.models import preprocess
    x = preprocess(x_u8, model.spec)
    with torch.inference_mode():
        out = model.backbone(pixel_values=x, output_attentions=True, output_hidden_states=True)
    atts = out.attentions
    if len(atts) != 12 or atts[0].shape[-1] != 197:
        raise AssertionError(f"expected 12 attention maps of 197 tokens, got {len(atts)}")
    eye = torch.eye(197, device=x.device)
    r = eye.expand(len(x), 197, 197)
    for a in atts:
        a = 0.5 * a.float().mean(1) + 0.5 * eye
        r = (a / a.sum(-1, keepdim=True)) @ r
    norms = out.hidden_states[-1][:, 1:].float().norm(dim=-1)
    outlier = (norms.max(1).values > 3 * norms.median(1).values).float()
    return r[:, 0, 1:].reshape(-1, 14, 14), outlier


def gradcam(model, x_u8):
    """(B, 8, 8) Grad-CAM for the 'fake' logit on EfficientNet's last feature map (fp32)."""
    import torch
    import torch.nn.functional as F
    from ekyc.models import preprocess
    x = preprocess(x_u8, model.spec)
    with torch.enable_grad():
        a = model.net.forward_features(x)
        logit = model.net.forward_head(a)
        (g,) = torch.autograd.grad(logit.sum(), a)
    return F.relu((g.mean((2, 3), keepdim=True) * a).sum(1)).detach()


def maps_for(model, kind, paths, is512, batch=32):
    """Normalised 256 x 256 maps (sum 1) for the given frames, and the ViT outlier flags."""
    import torch
    import torch.nn.functional as F
    from ekyc.data import decode_256
    maps, flags = [], []
    for i in range(0, len(paths), batch):
        x = torch.stack([decode_256(str(DATA / p), b) for p, b in zip(paths[i:i + batch], is512[i:i + batch])]).cuda()
        if kind == "clip_lora":
            m, f = rollout(model, x)
            flags.append(f.cpu())
        else:
            m = gradcam(model, x)
        m = F.interpolate(m[:, None].float(), size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        m = m.clamp_min(0)
        maps.append((m / m.sum((1, 2), keepdim=True).clamp_min(1e-12)).cpu())
    return torch.cat(maps).numpy(), (torch.cat(flags).numpy() if flags else None)


def stretch(m):
    """Display scaling: each map from its own minimum to its maximum (rollout maps are diffuse, so dividing by the
    maximum alone leaves almost no contrast)."""
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else m * 0


def ellipse_mask():
    import numpy as np
    cx, cy, ax, ay = ELLIPSE
    yy, xx = np.mgrid[0:256, 0:256]
    return ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1


def main():
    ap = argparse.ArgumentParser(description="Attention rollout (ViT) and Grad-CAM (EfficientNet) maps (step 5).")
    ap.add_argument("--scores", type=Path, default=OUT / "scores")
    ap.add_argument("--checkpoints", type=Path, default=OUT / "checkpoints")
    ap.add_argument("--results", type=Path, default=RESULTS, help="where analyze.py wrote thresholds.csv")
    ap.add_argument("--n", type=int, default=N_MEAN, help="frames per type and outcome for the mean maps")
    args = ap.parse_args()
    setup_console()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Ellipse
    from ekyc.data import decode_256, load_index
    from ekyc.models import load_checkpoint, set_eager_attention

    idx = load_index(columns=["path", "set", "method", "label", "src_res"])
    thr = pd.read_csv(args.results / "thresholds.csv").query("target_bpcer == 0.01").set_index("run")["threshold"]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ell = ellipse_mask()
    models, scores = {}, {}
    for kind in ("clip_lora", "effnet_b4"):
        run = f"{kind}_s0"
        models[kind], _ = load_checkpoint(args.checkpoints / run / "best.safetensors")
        z = np.load(args.scores / run / "scores.npz")
        scores[kind] = pd.Series(z["logit"][:, 0], index=z["row"])
    set_eager_attention(models["clip_lora"])
    rng = np.random.default_rng(SEED)

    rows, mean_maps, examples = [], {}, {}
    for t, methods in TYPES.items():
        fakes = idx[(idx["set"] == "df40_test") & idx["method"].isin(methods)]
        fakes = fakes[fakes.index.isin(scores["clip_lora"].index)]
        for kind in models:
            caught = scores[kind].loc[fakes.index].to_numpy() > thr[f"{kind}_s0"]
            for outcome, sel in (("caught", caught), ("missed", ~caught)):
                pick = fakes.index.to_numpy()[sel]
                pick = rng.choice(pick, min(args.n, len(pick)), replace=False) if len(pick) else pick
                if not len(pick):
                    continue
                m, flags = maps_for(models[kind], kind, idx.loc[pick, "path"].tolist(),
                                    (idx.loc[pick, "src_res"] == 512).tolist())
                mean_maps[(kind, t, outcome)] = m.mean(0)
                rows.append({"model": kind, "type": t, "outcome": outcome, "frames": len(pick),
                             "share_in_central_ellipse": float((m * ell).sum((1, 2)).mean()),
                             "vit_high_norm_token_share": float(flags.mean()) if flags is not None else None})
        vit_caught = scores["clip_lora"].loc[fakes.index].to_numpy() > thr["clip_lora_s0"]
        for outcome, sel in (("caught", vit_caught), ("missed", ~vit_caught)):
            if sel.any():
                examples[(t, outcome)] = int(rng.choice(fakes.index.to_numpy()[sel]))
    reals = idx[(idx["set"] == "cdf_test") & (idx["label"] == 0)]
    reals = reals[reals.index.isin(scores["clip_lora"].index)]
    rejected = reals.index.to_numpy()[scores["clip_lora"].loc[reals.index].to_numpy() > thr["clip_lora_s0"]]
    if len(rejected):
        examples[("genuine", "rejected")] = int(rng.choice(rejected))
    atomic_write(RESULTS / "explain_maps.csv" if args.results == RESULTS else args.results / "explain_maps.csv",
                 pd.DataFrame(rows).to_csv(index=False, float_format="%.4g", lineterminator="\n"))

    # Face-free figure: mean maps, rows = attack types, columns = model x outcome
    cols = [(k, o) for k in ("clip_lora", "effnet_b4") for o in ("caught", "missed")]
    titles = {"clip_lora": "ViT rollout", "effnet_b4": "EfficientNet Grad-CAM"}
    fig, axes = plt.subplots(3, 4, figsize=(8, 6.6))
    for r, t in enumerate(TYPES):
        for c, (k, o) in enumerate(cols):
            ax = axes[r, c]
            m = mean_maps.get((k, t, o))
            if m is not None:
                ax.imshow(stretch(m), cmap=cmap, vmin=0, vmax=1)
                n = next(x["frames"] for x in rows if (x["model"], x["type"], x["outcome"]) == (k, t, o))
                ax.set_xlabel(f"n = {n}", fontsize=7, color=INK2, labelpad=2)
            ax.add_patch(Ellipse(ELLIPSE[:2], 2 * ELLIPSE[2], 2 * ELLIPSE[3], fill=False, lw=0.8, ec=MUTED))
            ax.set_xlim(0, 256)
            ax.set_ylim(256, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(HAIRLINE)
            if r == 0:
                ax.set_title(f"{titles[k]}\n{o} fakes", fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(TYPE_NAMES[t], fontsize=9, color=INK)
    fig.suptitle("Where the models look: mean maps over DF40 fakes (seed 0, PNG)", fontsize=10, color=INK, y=0.995)
    fig.tight_layout()
    fig_path = OUT / "figures" / "explain_mean_maps.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, facecolor="white")
    plt.close(fig)
    register_figure(fig_path)

    # Faces figure (private): image | ViT map | EfficientNet map, one row per example
    keys = [k for k in [(t, o) for t in TYPES for o in ("caught", "missed")] + [("genuine", "rejected")]
            if k in examples]
    fig, axes = plt.subplots(len(keys), 3, figsize=(6.2, 2.1 * len(keys)), squeeze=False)
    for r, key in enumerate(keys):
        row = examples[key]
        img = decode_256(str(DATA / idx.at[row, "path"]), idx.at[row, "src_res"] == 512)
        mv, _ = maps_for(models["clip_lora"], "clip_lora", [idx.at[row, "path"]], [idx.at[row, "src_res"] == 512])
        mc, _ = maps_for(models["effnet_b4"], "effnet_b4", [idx.at[row, "path"]], [idx.at[row, "src_res"] == 512])
        gray = img.float().mean(0).numpy() / 255
        decisions = {k: ("fake" if scores[k].loc[row] > thr[f"{k}_s0"] else "genuine") for k in models}
        label = (f"{TYPE_NAMES.get(key[0], 'Genuine face')} ({idx.at[row, 'method']})"
                 if key[0] != "genuine" else "Genuine Celeb-DF face")
        for c, (panel, title) in enumerate(((None, label), (mv[0], f"ViT: {decisions['clip_lora']}"),
                                            (mc[0], f"EfficientNet: {decisions['effnet_b4']}"))):
            ax = axes[r, c]
            if panel is None:
                ax.imshow(img.permute(1, 2, 0).numpy())
            else:
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1, alpha=0.45)
                ax.imshow(stretch(panel), cmap=cmap, vmin=0, vmax=1, alpha=0.75)
            ax.set_title(title, fontsize=8, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.tight_layout()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAMPLES / "explain_grid.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[ OK ] {len(rows)} map groups; outputs/figures/explain_mean_maps.png (no faces), "
          "outputs/samples/explain_grid.png (faces, private), explain_maps.csv")


if __name__ == "__main__":
    main()
