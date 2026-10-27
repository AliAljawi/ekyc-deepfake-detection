"""
evaluate.py - scores one model on FF++ val (original PNG) and on every test frame (FF++ test, Celeb-DF test, the six
DF40 methods) at six levels: the original PNG and JPEG quality 90, 75, 60, 45, 30, applied to the 256x256 crop, to
real and fake frames alike.

Writes to the output folder: scores.npz (test rows + logits [n, 6], val rows + logits, the levels, and the index and
checkpoint fingerprints); scores.json (run facts and check results); evaluate.log; done.json last. No thresholds or
metrics are computed here: analyze.py derives the threshold from the stored val logits (a single source of truth).

Checks: the val AUC matches the training log (within 1e-4), and for 512 val frames the PNG logits from the 6-level
test batches match the plain val batches (max |difference| < 0.01, same decisions at the 1% threshold).

Usage (VS Code terminal, inside the code folder):
    .venv\\Scripts\\python.exe -B evaluate.py --ckpt outputs/checkpoints/clip_lora_s0/best.safetensors --out outputs/scores/clip_lora_s0
    .venv\\Scripts\\python.exe -B evaluate.py --zeroshot --out outputs/scores/clip_zeroshot
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import CACHE, JPEG_Q, LEVELS, NW, OUT  # noqa: E402
from ekyc.utils import (atomic_write, git_commit, keep_awake, retry_fs, setup_console, setup_logging,  # noqa: E402
                        versions)

TEST_SETS = ("ffpp_test", "cdf_test", "df40_test")
IMAGES_PER_BATCH = 384  # val: 384 frames x 1 level; test: 64 frames x 6 levels (the same forward-pass shape)


def score(model, ds, levels, log, what):
    """(rows, logits [n, levels], seconds) for every frame of ds, in index order."""
    import numpy as np
    import torch
    from ekyc.data import make_loader
    dl = make_loader(ds, batch_size=IMAGES_PER_BATCH // levels, num_workers=NW["eval"], seed=0)
    rows, logits = [], []
    t0, done, next_report = time.time(), 0, 0.25
    with torch.inference_mode():
        for x, r in dl:
            b = x.shape[0]
            logits.append(model(x.cuda(non_blocking=True).flatten(0, 1)).float().view(b, levels).cpu())
            rows.append(r)
            done += b
            if len(ds) > 20_000 and done / len(ds) >= next_report:
                rate = done * levels / (time.time() - t0)
                log.info(f"       {what}: {done / len(ds):4.0%} ({rate:,.0f} images/s)")
                next_report += 0.25
    return torch.cat(rows).numpy(), torch.cat(logits).numpy(), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Score one model on FF++ val and every test frame at 6 levels (step 4).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path, help="a trained model's best.safetensors")
    src.add_argument("--zeroshot", action="store_true", help="zero-shot CLIP with the pre-registered prompts")
    ap.add_argument("--out", type=Path, required=True, help="output folder (under outputs/)")
    ap.add_argument("--limit", type=int, help=argparse.SUPPRESS)  # testing: random frames per set
    args = ap.parse_args()
    setup_console()

    out = args.out.resolve()
    if OUT.resolve() not in out.parents:
        sys.exit(f"[FAIL] the output folder must be inside {OUT}")
    if (out / "done.json").exists():
        print(f"[ OK ] {out.name} already scored; nothing to do")
        return
    if out.exists():
        shutil.rmtree(out)  # an unfinished earlier attempt
    out.mkdir(parents=True)
    log = setup_logging(out / "evaluate.log")

    import numpy as np
    import pandas as pd
    import torch
    from ekyc.data import FrameDataset, index_sha256, load_index
    from ekyc.metrics import auc, threshold_at_bpcer
    from ekyc.models import ZeroShotCLIP, load_checkpoint

    with keep_awake():
        t_start = time.time()
        torch.backends.cudnn.benchmark = True
        sha = index_sha256()
        idx = load_index(columns=["path", "set", "method", "label", "src_res"])
        val = idx[idx["set"] == "ffpp_val"]
        test = idx[idx["set"].isin(TEST_SETS)]
        if args.limit:
            val = val.sample(n=min(args.limit, len(val)), random_state=0).sort_index()
            test = pd.concat([g.sample(n=min(args.limit, len(g)), random_state=0)
                              for _, g in test.groupby("set", observed=True)]).sort_index()

        if args.zeroshot:
            model = ZeroShotCLIP(CACHE).cuda().eval()
            facts, ckpt_sha, trained = {"kind": "clip_zeroshot", "prompts": model.prompts,
                                        "logit_scale": model.logit_scale}, "", None
        else:
            model, facts = load_checkpoint(args.ckpt)
            ckpt_sha = hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
            trained = json.loads((args.ckpt.parent / "result.json").read_text(encoding="utf-8"))
            if facts["index_sha256"] != sha:
                raise AssertionError("the checkpoint was trained on a different master list (index fingerprint differs)")
        log.info(f"Scoring {facts['kind']}" + (f" seed {facts['seed']}" if "seed" in facts else "")
                 + f": {len(val):,} val frames, {len(test):,} test frames x {len(LEVELS)} levels")

        val_rows, val_logits, val_s = score(model, FrameDataset.from_index(val, train=False), 1, log, "val")
        val_logits = val_logits[:, 0]
        y_val = val.loc[val_rows, "label"].to_numpy()
        val_auc = auc(val_logits, y_val)
        checks = {"val_auc": val_auc}
        if trained is not None and trained["val_frames"] == len(val):
            checks["val_auc_training"] = trained["val_auc"]
            if abs(val_auc - trained["val_auc"]) > 1e-4:
                raise AssertionError(f"val AUC {val_auc:.5f} differs from training's {trained['val_auc']:.5f}")
            log.info(f"[ OK ] val AUC {val_auc:.4f} matches the training log ({trained['val_auc']:.4f})")

        sub = val.sample(n=min(512, len(val)), random_state=1).sort_index()
        sub_rows, sub_logits, _ = score(model, FrameDataset.from_index(sub, train=False, jpeg_q=JPEG_Q), len(LEVELS),
                                        log, "check")
        ref = pd.Series(val_logits, index=val_rows).loc[sub_rows].to_numpy()
        t = threshold_at_bpcer(val_logits[y_val == 0])
        diff = float(np.abs(sub_logits[:, 0] - ref).max())
        flips = int(((sub_logits[:, 0] > t) != (ref > t)).sum())
        checks.update({"png_path_max_abs_diff": diff, "png_path_decision_flips": flips})
        if diff > 1e-2 or flips:
            raise AssertionError(f"PNG logits differ between the val and test paths: max |diff| {diff:.2e}, "
                                 f"{flips} decision flips")
        log.info(f"[ OK ] PNG logits agree between the val and test batch layouts (max |diff| {diff:.1e}, no flips)")

        test_rows, test_logits, test_s = score(model, FrameDataset.from_index(test, train=False, jpeg_q=JPEG_Q),
                                               len(LEVELS), log, "test")
        if not (np.isfinite(test_logits).all() and np.isfinite(val_logits).all()):
            raise FloatingPointError("non-finite logits")

        tmp = out / "scores.tmp.npz"
        np.savez(tmp, row=test_rows.astype(np.int32), logit=test_logits.astype(np.float32),
                 val_row=val_rows.astype(np.int32), val_logit=val_logits.astype(np.float32),
                 levels=np.array(LEVELS), index_sha256=np.array(sha), ckpt_sha256=np.array(ckpt_sha))
        retry_fs(os.replace, tmp, out / "scores.npz")
        record = {**{k: v for k, v in facts.items() if k != "versions"}, "checkpoint": str(args.ckpt or ""),
                  "ckpt_sha256": ckpt_sha, "index_sha256": sha, "levels": list(LEVELS), "val_frames": len(val),
                  "test_frames": len(test), "limited": bool(args.limit), **checks,
                  "val_seconds": val_s, "test_seconds": test_s,
                  "test_images_per_s": len(test) * len(LEVELS) / test_s,
                  "minutes": (time.time() - t_start) / 60, "scored_with_commit": git_commit(),
                  "scored_with_versions": versions(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        atomic_write(out / "scores.json", json.dumps(record, indent=1))
        atomic_write(out / "done.json", json.dumps({"finished": record["finished"]}))
        log.info(f"[ OK ] scored {len(test):,} test frames x {len(LEVELS)} levels in {test_s / 60:.1f} min "
                 f"({record['test_images_per_s']:,.0f} images/s); total {record['minutes']:.1f} min")


if __name__ == "__main__":
    main()
