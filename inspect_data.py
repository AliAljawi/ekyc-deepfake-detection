"""
inspect_data.py - step 1 checks on the images themselves (run after build_index.py).

  1. Sample faces: a 4x4 grid per group (16 different videos, fixed random draw), one overview sheet, and the mean
     face of 500 random frames per group -> outputs/samples/ (gitignored: they show faces)
  2. Simple image statistics on 500 random frames per group: sharpness (Laplacian variance), share of
     high-frequency energy, PNG bytes per pixel, 8x8 blockiness -> outputs/results/data_stats.csv
  3. One-feature shortcut check: how well each statistic alone separates each fake group from its real group (AUC).
     |AUC - 0.5| > 0.3 would mean a model could partly "cheat" on that set -> outputs/results/shortcut_auc.csv
  4. Loading speed with 0/4/8/12 DataLoader workers (training loader with augmentation, and the 6-level JPEG test
     loader), plus checks that every model gets identical augmented images -> outputs/results/loader_speed.csv

The statistics are computed on what the models see: 256x256 RGB, with SD-2.1 resized from 512 first.

Usage (VS Code terminal, inside the code folder):
    .venv\\Scripts\\python.exe -B inspect_data.py               # everything (~2 min)
    .venv\\Scripts\\python.exe -B inspect_data.py --no-speed    # skip the loader speed test
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import DATA, EPOCH_COMPOSITION, JPEG_Q, LOGS, RESULTS, SAMPLES  # noqa: E402
from ekyc.utils import atomic_write, setup_console, setup_logging  # noqa: E402

SEED = 2026
N_STATS = 500
GROUPS = [  # (key, label, set, method, the real group it's compared with)
    ("ffpp_real", "FF++ train: real", "ffpp_train", "real", None),
    ("ffpp_DF", "FF++ train: Deepfakes", "ffpp_train", "DF", "ffpp_real"),
    ("ffpp_F2F", "FF++ train: Face2Face", "ffpp_train", "F2F", "ffpp_real"),
    ("ffpp_FS", "FF++ train: FaceSwap", "ffpp_train", "FS", "ffpp_real"),
    ("ffpp_NT", "FF++ train: NeuralTextures", "ffpp_train", "NT", "ffpp_real"),
    ("cdf_real", "Celeb-DF test: real", "cdf_test", "real", None),
    ("cdf_fake", "Celeb-DF test: fake", "cdf_test", "Celeb-synthesis", "cdf_real"),
    ("df40_inswap", "DF40: InSwapper", "df40_test", "inswap", "cdf_real"),
    ("df40_simswap", "DF40: SimSwap", "df40_test", "simswap", "cdf_real"),
    ("df40_fomm", "DF40: FOMM", "df40_test", "fomm", "cdf_real"),
    ("df40_sadtalker", "DF40: SadTalker", "df40_test", "sadtalker", "cdf_real"),
    ("df40_sd2.1", "DF40: SD-2.1", "df40_test", "sd2.1", "cdf_real"),
    ("df40_DiT", "DF40: DiT-XL/2", "df40_test", "DiT", "cdf_real"),
]
STATS = ("laplacian_var", "hf_share", "png_bytes_per_px", "blockiness_8x8")
SHORTCUT_FLAG = 0.3


def _group_stats(job):
    """Runs in a worker process: statistics and the pixel sum (for the mean face) of one group's sample."""
    import numpy as np
    from torchvision.io import encode_png
    from ekyc.data import decode_256
    root, rels, is512 = job
    f = np.fft.fftfreq(256)
    high = np.hypot(f[:, None], f[None, :]) > 0.25  # above half the Nyquist frequency (0.5 cycles/pixel)
    out, acc = np.zeros((len(rels), len(STATS))), np.zeros((3, 256, 256))
    for k, (rel, big) in enumerate(zip(rels, is512)):
        img = decode_256(os.path.join(root, rel), big)
        x = img.numpy().astype(np.float64)
        acc += x
        y = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        lap = y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] - 4 * y[1:-1, 1:-1]
        power = np.abs(np.fft.fft2(y - y.mean())) ** 2
        dh, dv = np.abs(np.diff(y, axis=1)), np.abs(np.diff(y, axis=0))
        on_grid = dh[:, 7::8].mean() + dv[7::8, :].mean()  # differences across 8x8 block edges
        off_grid = np.delete(dh, np.s_[7::8], axis=1).mean() + np.delete(dv, np.s_[7::8], axis=0).mean()
        out[k] = (lap.var(), power[high].sum() / power.sum(), len(encode_png(img, compression_level=6)) / 256**2,
                  on_grid / off_grid)
    return out, acc


def draw_samples(idx):
    """Per group: 500 random frames for the statistics, and 16 frames from 16 different videos for the grid."""
    import numpy as np
    samples = {}
    for gi, (key, _, set_, method, _) in enumerate(GROUPS):
        rng = np.random.default_rng([SEED, gi])
        g = idx[(idx["set"] == set_) & (idx["method"] == method)]
        stats_rows = rng.choice(g.index.to_numpy(), N_STATS, replace=False)
        videos = rng.choice(np.asarray(g["video"].unique(), dtype=object), 16, replace=False)
        grid_rows = [rng.choice(g.index[g["video"] == v].to_numpy()) for v in videos]
        samples[key] = (np.sort(stats_rows), grid_rows)
    return samples


def faces(idx, samples, means, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torchvision.io import write_png
    from torchvision.utils import make_grid
    from ekyc.data import decode_256

    (SAMPLES / "grids").mkdir(parents=True, exist_ok=True)
    thumbs = {}
    for key, *_ in GROUPS:
        imgs = [decode_256(str(DATA / idx.at[r, "path"]), idx.at[r, "src_res"] == 512) for r in samples[key][1]]
        write_png(make_grid(torch.stack(imgs), nrow=4, padding=4, pad_value=255), str(SAMPLES / "grids" / f"{key}.png"))
        thumbs[key] = imgs[:8]

    fig, axes = plt.subplots(len(GROUPS), 9, figsize=(13, 1.45 * len(GROUPS)))
    for r, (key, label, *_) in enumerate(GROUPS):
        for c, img in enumerate(thumbs[key] + [means[key]]):
            ax = axes[r, c]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(c == 8)
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=10)
    axes[0, 8].set_title(f"mean of {N_STATS}", fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.995, top=0.985, bottom=0.005, wspace=0.04, hspace=0.06)
    fig.savefig(SAMPLES / "overview.png", dpi=110)
    plt.close(fig)
    log.info(f"[ OK ] sample faces -> outputs/samples/overview.png and outputs/samples/grids/<group>.png "
             f"({len(GROUPS)} groups, 16 videos each)")


def image_stats(idx, samples, log):
    import numpy as np
    import pandas as pd
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from sklearn.metrics import roc_auc_score

    jobs = [(str(DATA), idx.loc[samples[key][0], "path"].tolist(), (idx.loc[samples[key][0], "src_res"] == 512).tolist())
            for key, *_ in GROUPS]
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        results = dict(zip([g[0] for g in GROUPS], ex.map(_group_stats, jobs)))
    log.info(f"[ OK ] statistics on {N_STATS} frames x {len(GROUPS)} groups in {time.time() - t0:.0f} s")

    lines = []
    for key, label, *_ in GROUPS:
        vals = results[key][0]
        for j, stat in enumerate(STATS):
            v = vals[:, j]
            lines.append({"group": key, "label": label, "statistic": stat, "n": len(v), "mean": v.mean(),
                          "sd": v.std(ddof=1), "p10": np.percentile(v, 10), "median": np.median(v),
                          "p90": np.percentile(v, 90)})
    stats = pd.DataFrame(lines)
    atomic_write(RESULTS / "data_stats.csv", stats.to_csv(index=False, float_format="%.6g", lineterminator="\n"))

    rows = []
    for key, label, _, _, ref in GROUPS:
        if ref is None:
            continue
        x = np.concatenate([results[ref][0], results[key][0]])
        y = np.r_[np.zeros(N_STATS), np.ones(N_STATS)]
        aucs = {stat: roc_auc_score(y, x[:, j]) for j, stat in enumerate(STATS)}
        best = max(aucs, key=lambda s: abs(aucs[s] - 0.5))
        rows.append({"fake_group": key, "label": label, "vs": ref, **{f"auc_{s}": a for s, a in aucs.items()},
                     "best_statistic": best, "best_auc": aucs[best],
                     "flag": abs(aucs[best] - 0.5) > SHORTCUT_FLAG})
    shortcut = pd.DataFrame(rows)
    atomic_write(RESULTS / "shortcut_auc.csv", shortcut.to_csv(index=False, float_format="%.4f", lineterminator="\n"))

    wide = stats.pivot(index="label", columns="statistic", values="median").reindex([g[1] for g in GROUPS])
    log.info("Medians per group (500 frames each):\n" + wide[list(STATS)].to_string(float_format="%.4g"))
    log.info("One-feature shortcut check (AUC of each statistic alone; fake = positive):\n"
             + shortcut.drop(columns=["fake_group", "vs"]).to_string(index=False, float_format="%.3f"))
    flagged = shortcut.loc[shortcut["flag"], "label"].tolist()
    log.info(f"       flagged (|AUC - 0.5| > {SHORTCUT_FLAG}): {', '.join(flagged) if flagged else 'none'}")

    means = {key: torch.from_numpy(results[key][1] / N_STATS).round().clamp(0, 255).to(torch.uint8)
             for key, *_ in GROUPS}
    return means


def time_loader(dl, n_batches):
    """Seconds to the first batch (worker start-up) and images/s over the next n_batches."""
    t0 = time.time()
    it = iter(dl)
    first = next(it)
    t1 = time.time()
    n = 0
    for _ in range(n_batches):
        x, _ = next(it)
        n += x.shape[0] * (x.shape[1] if x.dim() == 5 else 1)
    return first, t1 - t0, n / (time.time() - t1)


def speed_test(idx, log):
    import pandas as pd
    import torch
    from ekyc.data import BalancedEpochSampler, FrameDataset, make_loader

    tr = idx[idx["set"] == "ffpp_train"]
    ds = FrameDataset.from_index(tr, train=True)
    sampler = BalancedEpochSampler.from_index(tr, seed=0)
    keep, seeds = sampler.draws(0)
    comp = tr["method"].iloc[keep].value_counts()
    comp = {m: int(comp.get(m, 0)) for m in EPOCH_COMPOSITION}
    if comp != EPOCH_COMPOSITION or len(keep) != len(sampler) or len(set(keep.tolist())) != len(keep):
        raise AssertionError(f"epoch 0 composition {comp} (len {len(keep)}), expected {EPOCH_COMPOSITION}")
    log.info(f"[ OK ] one training epoch = {len(keep):,} frames: {comp}")

    direct = torch.stack([ds[(int(keep[p]), int(seeds[p]))][0] for p in range(64)])
    again = torch.stack([ds[(int(keep[p]), int(seeds[p]))][0] for p in range(64)])
    if not torch.equal(direct, again):
        raise AssertionError("the same (frame, aug_seed) gave different augmented images")

    out = []
    for nw in (0, 4, 8, 12):
        dl = make_loader(ds, batch_size=64, num_workers=nw, seed=0, sampler=sampler, drop_last=True,
                         prefetch_factor=4)
        (x, rows), start, rate = time_loader(dl, 20 if nw == 0 else 100)
        if not (torch.equal(x, direct) and torch.equal(rows, torch.as_tensor(tr.index.to_numpy()[keep[:64]]))):
            raise AssertionError(f"{nw} workers: the first batch differs from the direct draw (augmentation "
                                 "must not depend on the worker)")
        out.append({"loader": "train (augmented, 1 image per frame)", "workers": nw, "first_batch_s": start,
                    "img_per_s": rate})
        log.info(f"       train loader, {nw:2d} workers: {rate:6,.0f} img/s (first batch after {start:.1f} s)")
        del dl
    log.info("[ OK ] identical augmented batches with 0, 4, 8 and 12 workers")

    test = idx[idx["set"] == "cdf_test"]
    ev = FrameDataset.from_index(test, train=False, jpeg_q=JPEG_Q)
    for nw in (4, 8, 12):
        dl = make_loader(ev, batch_size=64, num_workers=nw, seed=0)
        _, start, rate = time_loader(dl, 40)
        out.append({"loader": "test (6 JPEG levels per frame)", "workers": nw, "first_batch_s": start,
                    "img_per_s": rate})
        log.info(f"       test loader,  {nw:2d} workers: {rate:6,.0f} img/s = {rate / 6:,.0f} frames/s "
                 f"(first batch after {start:.1f} s)")
        del dl
    atomic_write(RESULTS / "loader_speed.csv",
                 pd.DataFrame(out).to_csv(index=False, float_format="%.1f", lineterminator="\n"))


def main():
    ap = argparse.ArgumentParser(description="Sample faces, image statistics and loader speed (step 1).")
    ap.add_argument("--no-speed", action="store_true", help="skip the loader speed test")
    ap.add_argument("--speed-only", action="store_true", help="only the loader speed test")
    args = ap.parse_args()
    setup_console()
    log = setup_logging(LOGS / "inspect_data.log")
    from ekyc.data import load_index
    t0 = time.time()
    idx = load_index()
    if not args.speed_only:
        samples = draw_samples(idx)
        means = image_stats(idx, samples, log)
        faces(idx, samples, means, log)
    if not args.no_speed:
        speed_test(idx, log)
    log.info(f"Done in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
