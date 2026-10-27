"""
train.py - trains one detector on FF++ train and picks its best epoch on FF++ val (never on a test set).

Each epoch: 23,012 frames (16 per real video, 4 per fake video), the same frames, order and augmentations for every
model with the same seed; AdamW, batch 64, 5% warm-up then cosine to 0, bf16. After every epoch: val AUC, balanced
val loss, and the in-domain AAR at 1% BPCER (monitoring only: the reported threshold comes from evaluate.py's val
scores via analyze.py). The best epoch follows the rule (val AUCs within 0.001 of the best tie ->
lowest balanced val loss -> earliest epoch); only its checkpoint is kept.

Writes to the run folder: train.log; history.json (numbers per epoch); best.safetensors + best.json (unless
--no-save); result.json; done.json last, so a crashed run is never mistaken for a finished one.

Usage (VS Code terminal, inside the code folder):
    .venv\\Scripts\\python.exe -B train.py --kind clip_lora --seed 0 --lr 3e-4 --out outputs/checkpoints/<run>
    .venv\\Scripts\\python.exe -B train.py --kind effnet_b4 --trial     # quick checks of one model (~2 min)
"""
import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import (BATCH, CLIP_NORM, EPOCH_COMPOSITION, EPOCHS, HEAD_LR, K_FAKE, K_REAL, LR_GRID,  # noqa: E402
                         NW, OUT, WARMUP_FRAC, WD)
from ekyc.utils import (atomic_write, git_commit, keep_awake, retry_fs, seed_everything,  # noqa: E402
                        setup_console, setup_logging, versions)

KINDS = ("clip_lora", "clip_probe", "effnet_b4")
INDEX_COLUMNS = ["path", "set", "method", "label", "video", "src_res"]
NONFINITE_EXIT = 3  # exit code for a non-finite loss
VAL_KEYS = ("val_auc", "val_loss_bal", "val_loss", "val_aar_at_1pct", "val_n")


def lr_factor(step, warmup, total):
    """Linear warm-up to the full rate, then cosine down to 0 at the last step."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def draws_hash(rows, keep, seeds):
    """Fingerprint of one epoch's frames, order and augmentation seeds; must be equal for all models of a seed."""
    h = hashlib.sha256(rows[keep].astype("int64").tobytes())
    h.update(seeds.astype("uint32").tobytes())
    return h.hexdigest()[:16]


def loaders(idx, seed, val_limit=None):
    from ekyc.data import BalancedEpochSampler, FrameDataset, make_loader
    tr = idx[idx["set"] == "ffpp_train"]
    va = idx[idx["set"] == "ffpp_val"]
    if val_limit:  # trials only
        va = va.sample(n=val_limit, random_state=0).sort_index()
    sampler = BalancedEpochSampler.from_index(tr, seed=seed)
    dl_tr = make_loader(FrameDataset.from_index(tr, train=True), batch_size=BATCH, num_workers=NW["train"],
                        seed=seed, sampler=sampler, drop_last=True, prefetch_factor=4)
    dl_va = make_loader(FrameDataset.from_index(va, train=False), batch_size=256, num_workers=NW["val"], seed=seed)
    return tr, va, sampler, dl_tr, dl_va


def predict(model, dl):
    """Index rows and logits for every frame of an evaluation loader (one JPEG level: the original PNG)."""
    import torch
    model.eval()
    rows, logits = [], []
    with torch.inference_mode():
        for x, r in dl:
            logits.append(model(x.cuda(non_blocking=True).flatten(0, 1)).float().cpu())
            rows.append(r)
    return torch.cat(rows).numpy(), torch.cat(logits).numpy()


def val_metrics(va, rows, logits):
    from ekyc.metrics import aar, auc, balanced_bce, bce, threshold_at_bpcer
    sub = va.loc[rows]
    y, m = sub["label"].to_numpy(), sub["method"].astype(str).to_numpy()
    t = threshold_at_bpcer(logits[y == 0])
    return {"val_auc": auc(logits, y), "val_loss_bal": balanced_bce(logits, y, m), "val_loss": float(bce(logits, y).mean()),
            "val_aar_at_1pct": aar(logits[y == 1], t), "val_n": int(len(y))}


def train(kind, seed, lr, epochs, run_dir, log, save=True, max_steps=None, val_limit=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ekyc.data import index_sha256, load_index
    from ekyc.metrics import select
    from ekyc.models import build_model, save_checkpoint

    t_start = time.time()
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    idx = load_index(columns=INDEX_COLUMNS)
    tr, va, sampler, dl_tr, dl_va = loaders(idx, seed, val_limit)
    labels = torch.tensor(idx["label"].to_numpy(), dtype=torch.float32, device="cuda")
    model = build_model(kind, seed).cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(model.param_groups(lr, HEAD_LR, WD), fused=True)
    steps = len(sampler) // BATCH if max_steps is None else min(max_steps, len(sampler) // BATCH)
    total = epochs * steps
    warmup = max(1, round(WARMUP_FRAC * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, functools.partial(lr_factor, warmup=warmup, total=total))
    meta = {"kind": kind, "seed": seed, "lr": lr, "head_lr": HEAD_LR if kind != "clip_probe" else lr,
            "epochs": epochs, "batch": BATCH, "steps_per_epoch": steps, "warmup_steps": warmup, "total_steps": total,
            "weight_decay": WD, "grad_clip": CLIP_NORM, "k_real": K_REAL, "k_fake": K_FAKE,
            "trainable_params": model.n_trainable(), "total_params": sum(p.numel() for p in model.parameters()),
            "index_sha256": index_sha256(), "git_commit": git_commit(), "versions": versions(),
            "gpu": torch.cuda.get_device_name(0), "val_frames": len(va), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log.info(f"{kind} seed {seed} lr {lr:g}: {epochs} epochs x {steps} steps, warm-up {warmup} steps, "
             f"{meta['trainable_params']:,} of {meta['total_params']:,} parameters trainable")

    history, tr_rows = [], tr.index.to_numpy()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        keep, seeds = sampler.draws(epoch)
        comp = tr["method"].iloc[keep].value_counts()
        if {m: int(comp.get(m, 0)) for m in EPOCH_COMPOSITION} != EPOCH_COMPOSITION:
            raise AssertionError(f"epoch {epoch}: composition {dict(comp)} differs from {EPOCH_COMPOSITION}")
        model.train()
        t0 = time.time()
        loss_sum = torch.zeros((), device="cuda")
        gnorm_sum = torch.zeros((), device="cuda")
        n = 0
        for x, rows in dl_tr:
            if n == steps:
                break
            x = x.cuda(non_blocking=True)
            y = labels[rows.cuda(non_blocking=True)]
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm_sum += torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM)
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            n += 1
            # Checking every step would make the CPU wait for the GPU each time (EfficientNet: ~35% slower)
            if n % 50 == 0 and not torch.isfinite(gnorm_sum + loss_sum):
                raise FloatingPointError(f"epoch {epoch}, step {n}: non-finite loss or gradient")
        torch.cuda.synchronize()
        secs = time.time() - t0
        t1 = time.time()
        rows_v, logits_v = predict(model, dl_va)
        rec = {"epoch": epoch, "train_loss": loss_sum.item() / n, "grad_norm": gnorm_sum.item() / n,
               "lr_end": opt.param_groups[0]["lr"], "img_per_s": n * BATCH / secs,
               **val_metrics(va, rows_v, logits_v), "val_s": time.time() - t1,
               "draws": draws_hash(tr_rows, keep, seeds), "minutes": (time.time() - t_start) / 60}
        if not np.isfinite([rec["train_loss"], rec["val_loss"]]).all():
            raise FloatingPointError(f"epoch {epoch}: non-finite loss {rec}")
        history.append(rec)
        atomic_write(Path(run_dir) / "history.json", json.dumps(history, indent=1))
        if save:
            save_checkpoint(model, Path(run_dir) / f"epoch_{epoch:02d}.safetensors", {**meta, "epoch": epoch})
        log.info(f"epoch {epoch + 1:2d}/{epochs} | train loss {rec['train_loss']:.4f} | val AUC {rec['val_auc']:.4f} "
                 f"| val loss (bal.) {rec['val_loss_bal']:.4f} | val AAR@1% {rec['val_aar_at_1pct']:.3f} | "
                 f"{rec['img_per_s']:,.0f} img/s | {rec['minutes']:.1f} min")

    best = select(history)
    result = {**meta, "best_epoch": best, **{k: history[best][k] for k in VAL_KEYS},
              "train_minutes": (time.time() - t_start) / 60,
              "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3}
    if save:
        for e in range(epochs):
            f = Path(run_dir) / f"epoch_{e:02d}.safetensors"
            if e == best:
                retry_fs(os.replace, f, f.with_name("best.safetensors"))
                retry_fs(os.replace, f.with_suffix(".json"), f.with_name("best.json"))
            else:
                retry_fs(f.unlink)
                retry_fs(f.with_suffix(".json").unlink)
    atomic_write(Path(run_dir) / "result.json", json.dumps(result, indent=1))
    log.info(f"[ OK ] best epoch {best + 1} (val AUC {result['val_auc']:.4f}, balanced val loss "
             f"{result['val_loss_bal']:.4f}); {result['train_minutes']:.1f} min, peak GPU memory "
             f"{result['peak_gpu_gb']:.1f} GB")
    return result


def trial(kind, run_dir, log):
    """Quick checks before any long run: trainable parameters, frozen weights stay frozen, one batch can be learned
    by heart, a checkpoint reloads exactly, and a 2-epoch mini run works end to end (selection, saving, reloading)."""
    import torch
    import torch.nn.functional as F
    from ekyc.data import load_index
    from ekyc.models import build_model, load_checkpoint, save_checkpoint

    lr = LR_GRID[kind][1]
    torch.backends.cudnn.benchmark = True
    idx = load_index(columns=INDEX_COLUMNS)
    tr, va, sampler, dl_tr, dl_va = loaders(idx, seed=0, val_limit=2_000)
    labels = torch.tensor(idx["label"].to_numpy(), dtype=torch.float32, device="cuda")
    model = build_model(kind, 0).cuda()
    n_tr, n_all = model.n_trainable(), sum(p.numel() for p in model.parameters())
    if kind == "effnet_b4" and n_tr != n_all:
        raise AssertionError(f"EfficientNet should be fully trainable: {n_tr:,} of {n_all:,}")
    log.info(f"[ OK ] {kind}: {n_tr:,} of {n_all:,} parameters trainable")

    x, rows = next(iter(dl_tr))
    x, y = x.cuda(), labels[rows.cuda()]
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(model.param_groups(lr, HEAD_LR, WD), fused=True)
    model.train()
    t0 = time.time()
    for step in range(100):
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM, error_if_nonfinite=True)
        opt.step()
        if step == 0:
            first = loss.item()
        if step == 19 and frozen:
            changed = [n for n, p in model.named_parameters() if n in frozen and not torch.equal(p, frozen[n])]
            if changed:
                raise AssertionError(f"frozen CLIP weights changed: {changed[:3]}")
            log.info(f"[ OK ] all {len(frozen)} frozen CLIP tensors are bit-identical after 20 steps")
    final = loss.item()
    log.info(f"{'[ OK ]' if final < 0.05 else '[FAIL]'} one batch of 64 learned by heart: loss {first:.3f} -> "
             f"{final:.4f} in 100 steps at lr {lr:g} ({100 * 64 / (time.time() - t0):,.0f} img/s)")
    if final >= 0.05:
        raise AssertionError("the model can't fit a single batch: something is wrong with the training step")

    save_checkpoint(model, run_dir / "roundtrip.safetensors", {"seed": 0})
    reloaded, _ = load_checkpoint(run_dir / "roundtrip.safetensors")
    a, b = model.checkpoint_state(), reloaded.checkpoint_state()
    if a.keys() != b.keys() or not all(torch.equal(a[k], b[k]) for k in a):
        raise AssertionError("checkpoint round trip changed some tensors")
    model.eval()
    with torch.inference_mode():
        diff = (model(x) - reloaded(x)).abs().max().item()
    if diff > 1e-3:
        raise AssertionError(f"reloaded model gives different logits (max |diff| {diff:.2e})")
    log.info(f"[ OK ] checkpoint round trip: {len(a)} tensors identical, logits max |diff| {diff:.1e}")
    del model, reloaded, opt, frozen

    log.info("Mini run: 2 epochs x 40 steps, 2,000 val frames...")
    result = train(kind, 0, lr, 2, run_dir, log, save=True, max_steps=40, val_limit=2_000)
    best, meta = load_checkpoint(run_dir / "best.safetensors")
    from ekyc.metrics import auc
    rows_v, logits_v = predict(best, dl_va)
    again = auc(logits_v, va.loc[rows_v, "label"].to_numpy())
    if abs(again - result["val_auc"]) > 1e-4:
        raise AssertionError(f"reloaded best checkpoint: val AUC {again:.5f} vs {result['val_auc']:.5f} in training")
    log.info(f"[ OK ] reloaded best checkpoint (epoch {meta['epoch'] + 1}) reproduces its val AUC ({again:.4f})")
    log.info(f"[ OK ] trial passed for {kind}")


def main():
    ap = argparse.ArgumentParser(description="Train one detector on FF++ (step 3).")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, help="backbone/adapter learning rate (the probe: its head's)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", type=Path, help="run folder (under outputs/)")
    ap.add_argument("--no-save", action="store_true", help="don't keep a checkpoint (learning-rate sweep)")
    ap.add_argument("--trial", action="store_true", help="quick checks instead of a run")
    ap.add_argument("--max-steps", type=int, help=argparse.SUPPRESS)  # testing: steps per epoch
    ap.add_argument("--val-limit", type=int, help=argparse.SUPPRESS)  # testing: random val frames
    args = ap.parse_args()
    setup_console()
    if not args.trial and (args.out is None or args.lr is None):
        sys.exit("[FAIL] a run needs --out and --lr")

    run_dir = OUT / "checkpoints" / f"trial_{args.kind}" if args.trial else args.out.resolve()
    if OUT.resolve() not in run_dir.resolve().parents:
        sys.exit(f"[FAIL] the run folder must be inside {OUT}")
    if (run_dir / "done.json").exists():
        print(f"[ OK ] {run_dir.name} already finished; nothing to do")
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)  # a crashed or trial run: start over
    run_dir.mkdir(parents=True)
    log = setup_logging(run_dir / "train.log")
    with keep_awake():
        if args.trial:
            trial(args.kind, run_dir, log)
            return
        try:
            train(args.kind, args.seed, args.lr, args.epochs, run_dir, log, save=not args.no_save,
                  max_steps=args.max_steps, val_limit=args.val_limit)
        except FloatingPointError as e:
            log.error(f"[FAIL] non-finite loss or gradient: {e}")
            if not args.no_save:
                sys.exit(NONFINITE_EXIT)  # run_all.py stops on this code instead of retrying (pre-registered gate)
            # A learning-rate candidate in the sweep: recorded as diverged, so it can't be selected (deviation 4)
            atomic_write(run_dir / "result.json", json.dumps({"kind": args.kind, "seed": args.seed, "lr": args.lr,
                                                              "epochs": args.epochs, "diverged": True, "error": str(e),
                                                              "best_epoch": None, "val_auc": None,
                                                              "val_loss_bal": None, "train_minutes": None}, indent=1))
        atomic_write(run_dir / "done.json", json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}))


if __name__ == "__main__":
    main()
