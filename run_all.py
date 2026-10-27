"""
run_all.py - the whole experiment (steps 3-4) in one command, resumable.

Order: learning-rate sweep (seed 0, 4 epochs, 3 rates per model, plus one more rate if the best is at a grid edge)
-> rate choice by the pre-registered rule (outputs/results/lr_choice.json) -> gate (stop if a model's best sweep val
AUC < 0.60) -> train seed 0 (3 models x 20 epochs) -> score seed 0 and zero-shot CLIP -> train seeds 1-2 -> score them.

Every job runs as its own process and is retried once (never after a non-finite loss: that is the pre-registered gate).
A job whose done.json exists is skipped, so running this again after a crash or a restart continues where it stopped
(an unfinished job starts over); a skipped job must have been run with the same arguments, and scores must belong to
the current checkpoint. The code that jobs import must not change while this runs (it stops if it does). A lock file
stops two copies running at once, and the PC is kept awake. Everything is logged to outputs/logs/run_all.log.

Usage (VS Code terminal, inside the code folder):
    .venv\\Scripts\\python.exe -B run_all.py --stage sweep     # sweep + learning-rate choice only (~25 min)
    .venv\\Scripts\\python.exe -B run_all.py                   # everything, resuming (~3 h)
    .venv\\Scripts\\python.exe -B run_all.py --tiny            # rehearsal of the whole chain on a few batches
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import CODE, EPOCHS, LOGS, LR_GRID, OUT, RESULTS, SEEDS, SWEEP_EPOCHS  # noqa: E402
from ekyc.metrics import select  # noqa: E402
from ekyc.utils import atomic_write, keep_awake, setup_console  # noqa: E402

KINDS = ("clip_lora", "clip_probe", "effnet_b4")
GATE_AUC = 0.60
NONFINITE_EXIT = 3  # train.py's exit code for a non-finite loss
TINY_TRAIN = ["--max-steps", "6", "--val-limit", "300"]
TINY_EVAL = ["--limit", "400"]
JOB_CODE = ("train.py", "evaluate.py", "ekyc/__init__.py", "ekyc/config.py", "ekyc/data.py", "ekyc/models.py",
            "ekyc/metrics.py", "ekyc/utils.py")


class Log:
    """Timestamped lines to run_all.log (line-buffered, so the log can be watched live) and, best effort, the screen.
    The file is written first: a frozen or closed console must never hold up the run."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg, stamp=True):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}" if stamp else msg
        self.f.write(line + "\n")
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass


def disable_quick_edit():
    """A click inside a classic Windows console starts a 'QuickEdit' selection that blocks every write to it, which
    would silently freeze the run. Switch QuickEdit off for this console (no effect without a console)."""
    if os.name != "nt":
        return
    import ctypes
    k32 = ctypes.windll.kernel32
    handle = k32.GetStdHandle(-10)  # standard input
    mode = ctypes.c_uint32()
    if k32.GetConsoleMode(handle, ctypes.byref(mode)):
        k32.SetConsoleMode(handle, (mode.value | 0x80) & ~0x40)  # ENABLE_EXTENDED_FLAGS on, ENABLE_QUICK_EDIT_MODE off


def code_fingerprint():
    h = hashlib.sha256()
    for f in JOB_CODE:
        h.update(f.encode())
        h.update((CODE / f).read_bytes())
    return h.hexdigest()[:16]


def choose_lr(records):
    """records: [{lr, val_auc, val_loss_bal, ...}] for one model. Returns (chosen record, extra rate to try or None).
    The rule, with ties going to the lower rate; an extra rate x3 beyond the grid if the best is at an
    edge (only while the grid has its original 3 rates). Diverged candidates (non-finite loss) can't be chosen."""
    recs = sorted(records, key=lambda r: r["lr"])
    ok = [r for r in recs if not r.get("diverged")]
    if not ok:
        raise RuntimeError("gate: every learning rate in the sweep diverged")
    best = ok[select(ok)]
    extra = None
    if len(recs) == 3:
        if best["lr"] == recs[0]["lr"]:
            extra = recs[0]["lr"] / 3
        elif best["lr"] == recs[-1]["lr"]:
            extra = recs[-1]["lr"] * 3
    return best, extra


class Runner:
    def __init__(self, log, tiny):
        self.log, self.tiny = log, tiny
        self.ckpt = OUT / "checkpoints" / "tiny" if tiny else OUT / "checkpoints"
        self.scores = OUT / "scores" / "tiny" if tiny else OUT / "scores"
        self.results = self.ckpt if tiny else RESULTS
        self.seeds = SEEDS[:2] if tiny else SEEDS
        self.code = code_fingerprint()
        self.log(f"       job code fingerprint {self.code} ({', '.join(JOB_CODE)})")

    def job(self, name, args, folder):
        record = folder / "run_all_job.json"
        if (folder / "done.json").exists():
            if record.exists():
                old = json.loads(record.read_text(encoding="utf-8"))
                if old["args"] != args:
                    raise RuntimeError(f"{name}: {folder} was made with different arguments ({old['args']}); "
                                       "delete that folder to redo it with the current ones")
                note = "" if old["code"] == self.code else f" (made with job code {old['code']})"
            else:
                note = " (made before job records were kept)"
            self.log(f"[skip] {name}: already done{note}")
            return
        if code_fingerprint() != self.code:
            raise RuntimeError("the code the jobs import changed while run_all.py was running; restart run_all.py so "
                               "every remaining job uses one version")
        cmd = [sys.executable, "-B", *args]
        for attempt in (1, 2):
            self.log(f"[start] {name}" + (" (retry)" if attempt == 2 else ""))
            t0 = time.time()
            proc = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
            for line in proc.stdout:
                self.log(f"    {line.rstrip()}", stamp=False)
            rc = proc.wait()
            if rc == 0 and (folder / "done.json").exists():
                atomic_write(record, json.dumps({"args": args, "code": self.code,
                                                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
                self.log(f"[done] {name} in {(time.time() - t0) / 60:.1f} min")
                return
            self.log(f"[FAIL] {name}: exit code {rc}")
            if rc == NONFINITE_EXIT:  # a non-finite loss is the pre-registered gate, not a glitch: never retry it
                raise RuntimeError(f"gate: {name} hit a non-finite loss; stopping")
        raise RuntimeError(f"{name} failed twice; stopping. Fix the cause, then run run_all.py again to resume.")

    def train(self, name, kind, seed, lr, epochs, folder, save=True):
        args = ["train.py", "--kind", kind, "--seed", str(seed), "--lr", repr(lr), "--epochs", str(epochs),
                "--out", str(folder)]
        args += ([] if save else ["--no-save"]) + (TINY_TRAIN if self.tiny else [])
        self.job(name, args, folder)
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if save and result.get("diverged"):
            raise RuntimeError(f"gate: {name} diverged (non-finite loss)")
        return result

    def sweep(self):
        choice = {}
        for kind in KINDS:
            grid = LR_GRID[kind][1:2] if self.tiny else LR_GRID[kind]
            recs = []
            for lr in grid:
                recs.append(self.train(f"sweep {kind} lr {lr:g}", kind, 0, lr, 2 if self.tiny else SWEEP_EPOCHS,
                                       self.ckpt / "sweep" / f"{kind}_lr{lr:g}", save=False))
            best, extra = choose_lr(recs)
            if extra is not None and not self.tiny:
                self.log(f"       {kind}: best rate {best['lr']:g} is at the grid edge; trying {extra:g} as well")
                recs.append(self.train(f"sweep {kind} lr {extra:g}", kind, 0, extra, SWEEP_EPOCHS,
                                       self.ckpt / "sweep" / f"{kind}_lr{extra:g}", save=False))
                best, _ = choose_lr(recs)
            lrs = [r["lr"] for r in recs]
            choice[kind] = {"lr": best["lr"], "val_auc": best["val_auc"], "val_loss_bal": best["val_loss_bal"],
                            "best_epoch": best["best_epoch"], "extended": len(recs) > len(grid),
                            "inside_grid": min(lrs) < best["lr"] < max(lrs),
                            "runs": [{k: r.get(k) for k in ("lr", "best_epoch", "val_auc", "val_loss_bal",
                                                            "train_minutes", "diverged")}
                                     for r in sorted(recs, key=lambda r: r["lr"])]}
            self.log(f"[ OK ] {kind}: learning rate {best['lr']:g} (val AUC {best['val_auc']:.4f}, balanced val loss "
                     f"{best['val_loss_bal']:.4f}, epoch {best['best_epoch'] + 1})")
        atomic_write(self.results / "lr_choice.json", json.dumps(choice, indent=1))
        if not self.tiny:
            low = [k for k, c in choice.items() if c["val_auc"] < GATE_AUC]
            if low:
                raise RuntimeError(f"gate: best sweep val AUC below {GATE_AUC} for {low}; stopping before the main runs")
            self.log(f"[ OK ] gate passed: every model's best sweep val AUC >= {GATE_AUC}")
        self.check_draws([self.ckpt / "sweep" / f"{k}_lr{c['lr']:g}" for k, c in choice.items()], "sweep")
        return choice

    def check_draws(self, folders, what):
        """Every model trained with the same seed must have seen identical frames, order and augmentations."""
        draws = [[e["draws"] for e in json.loads((f / "history.json").read_text(encoding="utf-8"))] for f in folders]
        n = min(len(d) for d in draws)
        if any(d[:n] != draws[0][:n] for d in draws):
            raise RuntimeError(f"{what}: the models saw different training draws: {draws}")
        self.log(f"[ OK ] {what}: identical training frames and augmentations across the 3 models ({n} epochs)")

    def train_seed(self, seed, choice):
        folders = []
        for kind in KINDS:
            folder = self.ckpt / f"{kind}_s{seed}"
            self.train(f"train {kind} seed {seed}", kind, seed, choice[kind]["lr"], 2 if self.tiny else EPOCHS, folder)
            folders.append(folder)
        self.check_draws(folders, f"seed {seed}")

    def score(self, name, args, folder, ckpt=None):
        self.job(name, ["evaluate.py", *args, "--out", str(folder)] + (TINY_EVAL if self.tiny else []), folder)
        if ckpt is not None:
            stored = json.loads((folder / "scores.json").read_text(encoding="utf-8"))["ckpt_sha256"]
            if stored != hashlib.sha256(ckpt.read_bytes()).hexdigest():
                raise RuntimeError(f"{name}: {folder} holds scores of a different checkpoint; delete it to re-score")

    def score_seed(self, seed):
        for kind in KINDS:
            ckpt = self.ckpt / f"{kind}_s{seed}" / "best.safetensors"
            self.score(f"score {kind} seed {seed}", ["--ckpt", str(ckpt)], self.scores / f"{kind}_s{seed}", ckpt)

    def run(self, stage):
        choice = self.sweep()
        if stage == "sweep":
            return
        self.train_seed(self.seeds[0], choice)
        self.score_seed(self.seeds[0])
        self.score("score zero-shot CLIP", ["--zeroshot"], self.scores / "clip_zeroshot")
        for seed in self.seeds[1:]:
            self.train_seed(seed, choice)
        for seed in self.seeds[1:]:
            self.score_seed(seed)


def acquire_lock(lock, tiny):
    """One run at a time: refuse if the lock's owner is a live run_all.py, or if a training or scoring job from an
    earlier run of the same kind (real or rehearsal) is still alive (it would collide with the restarted job)."""
    import psutil

    def cmdline(p):
        try:
            return " ".join(p.cmdline())
        except (psutil.Error, OSError):
            return ""

    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").split()[0])
        except (ValueError, IndexError):
            pid = None
        if pid and pid != os.getpid() and psutil.pid_exists(pid) and "run_all.py" in cmdline(psutil.Process(pid)):
            sys.exit(f"[FAIL] run_all.py is already running (process {pid}); not starting a second copy")
    def same_kind(c):
        return str(OUT) in c and (("\\tiny\\" in c) or ("/tiny/" in c)) == tiny

    leftovers = [p.pid for p in psutil.process_iter(["name"]) if "python" in (p.info["name"] or "").lower()
                 and p.pid != os.getpid() and any(s in cmdline(p) for s in ("train.py", "evaluate.py"))
                 and same_kind(cmdline(p))]
    if leftovers:
        sys.exit(f"[FAIL] training/scoring processes from an earlier run are still alive: {leftovers}. Stop them "
                 "(taskkill /T /F /PID <pid>) and start again.")
    atomic_write(lock, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    ap = argparse.ArgumentParser(description="Run the experiment (steps 3-4), resuming where it stopped.")
    ap.add_argument("--stage", choices=("sweep", "all"), default="all")
    ap.add_argument("--tiny", action="store_true", help="rehearsal on a few batches (separate folders)")
    args = ap.parse_args()
    setup_console()
    disable_quick_edit()
    log = Log(LOGS / ("run_all_tiny.log" if args.tiny else "run_all.log"))
    lock = OUT / (".run_all_tiny.lock" if args.tiny else ".run_all.lock")
    acquire_lock(lock, args.tiny)
    t0 = time.time()
    try:
        with keep_awake():
            log(f"===== run_all.py --stage {args.stage}{' --tiny' if args.tiny else ''} (process {os.getpid()})")
            Runner(log, args.tiny).run(args.stage)
            log(f"===== ALL DONE ({args.stage}) in {(time.time() - t0) / 60:.0f} min")
    except BaseException as e:  # noqa: BLE001 - log every stop, including Ctrl+C
        log(f"===== STOPPED after {(time.time() - t0) / 60:.0f} min: {type(e).__name__}: {e}")
        log(traceback.format_exc(), stamp=False)
        raise SystemExit(1)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
