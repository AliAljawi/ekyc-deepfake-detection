"""
Small helpers shared by the scripts: console and log setup, keeping Windows awake, seeding, safe file writes,
and the facts every run records (git commit, library versions).
"""
import contextlib
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from ekyc.config import CODE


def setup_console():
    """UTF-8 console output on Windows (paths and names can contain non-ASCII characters)."""
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(log_path, name="ekyc"):
    """Logs to the screen and to log_path; uncaught errors are logged too, so a crash shows up in the file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)

    def excepthook(kind, value, tb):
        log.error("Uncaught error", exc_info=(kind, value, tb))

    sys.excepthook = excepthook
    return log


@contextlib.contextmanager
def keep_awake():
    """Stops Windows from sleeping while a long job runs (the screen may still turn off; locking is fine)."""
    if platform.system() != "Windows":
        yield
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def retry_fs(fn, *args, attempts=3, wait=1.0):
    """Runs a file operation, retrying on PermissionError: on Windows, antivirus or the search indexer can briefly
    lock a file that was just written."""
    for attempt in range(attempts):
        try:
            return fn(*args)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait)


def atomic_write(path, data):
    """Writes to a temporary file, flushes it to disk, then renames it, so a crash or power cut never leaves a
    half-written or empty file under the final name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data.encode("utf-8") if isinstance(data, str) else data)
        f.flush()
        os.fsync(f.fileno())
    retry_fs(os.replace, tmp, path)


def git_commit():
    """The current commit, with '+dirty' if tracked files have uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=CODE,
                               capture_output=True, text=True).stdout.strip()
        return (head or "no-commit") + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def versions():
    """Library versions for the run records."""
    out = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "transformers", "peft", "timm", "numpy", "pandas", "sklearn", "scipy"):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except ImportError:
                continue
        out[name] = getattr(mod, "__version__", "")
    return out


def run_name(kind, seed, lr, epochs):
    return f"{kind}_s{seed}_lr{lr:g}_e{epochs}"


def register_figure(path):
    """Adds a face-free figure in outputs/figures/ to MANIFEST.txt, the list check_repo.py allows on GitHub."""
    from ekyc.config import OUT
    figures = OUT / "figures"
    rel = Path(path).resolve().relative_to(figures.resolve()).as_posix()
    manifest = figures / "MANIFEST.txt"
    names = set(manifest.read_text(encoding="utf-8").split()) if manifest.is_file() else set()
    atomic_write(manifest, "\n".join(sorted(names | {rel})) + "\n")
