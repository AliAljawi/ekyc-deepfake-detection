"""
Settings shared by every script: paths, data sampling, training and evaluation constants.

Imports only os/pathlib, so entry scripts import it first: it sets the Hugging Face variables before any HF library
loads (huggingface_hub reads HF_HUB_OFFLINE once, at import time).
"""
import os
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # every model is cached: never touch the network during runs
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Paths
CODE = Path(__file__).resolve().parents[1]   # ...\code
ROOT = CODE.parent                            # C:\Applied Vision Transformer Report
DATA = ROOT / "datasets"                      # never search or glob inside it: JSON splits + targeted listings only
ZIPS = ROOT / "datasets_zip_folders"
OUT = CODE / "outputs"
INDEX = OUT / "index" / "frames.csv.gz"       # gitignored (labels derived from the datasets and annotations)
RESULTS = OUT / "results"                     # committed: tables of numbers only
CACHE = OUT / "cache"                         # gitignored
LOGS = OUT / "logs"
SAMPLES = OUT / "samples"                     # gitignored: anything that shows a face

# The master list (build_index.py)
SETS = ("ffpp_train", "ffpp_val", "ffpp_test", "cdf_test", "df40_test")
METHODS = ("real", "DF", "F2F", "FS", "NT", "Celeb-synthesis",
           "inswap", "simswap", "fomm", "sadtalker", "sd2.1", "DiT")
ATTACK_TYPE = {"real": "real", "DF": "faceswap", "FS": "faceswap", "F2F": "reenactment", "NT": "reenactment",
               "Celeb-synthesis": "faceswap", "inswap": "faceswap", "simswap": "faceswap",
               "fomm": "reenactment", "sadtalker": "reenactment", "sd2.1": "diffusion", "DiT": "diffusion"}
DF40_METHODS = ("inswap", "simswap", "fomm", "sadtalker", "sd2.1", "DiT")
ATTACK_TYPES = ("faceswap", "reenactment", "diffusion")

# Training data per epoch: k random frames per video (all if fewer), so real:fake = 1:1 and methods are equal
K_REAL, K_FAKE = 16, 4
EPOCH_COMPOSITION = {"real": 11_504, "DF": 2_876, "F2F": 2_876, "FS": 2_880, "NT": 2_876}  # = 23,012 frames

# Training
BATCH = 64
EPOCHS = 20
SWEEP_EPOCHS = 4
WARMUP_FRAC = 0.05
WD = 0.01
CLIP_NORM = 1.0
HEAD_LR = 1e-3
LR_GRID = {"clip_lora": (1e-4, 3e-4, 1e-3), "effnet_b4": (5e-5, 2e-4, 5e-4), "clip_probe": (1e-3, 3e-3, 1e-2)}
SEEDS = (0, 1, 2)
NW = {"train": 8, "val": 6, "eval": 8}  # DataLoader workers (inspect_data.py: 8 train workers ~3,500 img/s)

# Evaluation: original PNG plus five JPEG qualities, applied to the 256x256 crop
JPEG_Q = (None, 90, 75, 60, 45, 30)
LEVELS = ("png", "q90", "q75", "q60", "q45", "q30")
TARGET_BPCER = 0.01

# Models
CLIP_ID = "openai/clip-vit-base-patch16"
EFFNET_ID = "efficientnet_b4.ra2_in1k"
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InputSpec(NamedTuple):
    size: int
    mean: tuple
    std: tuple
    channels_last: bool


SPEC = {
    "clip_lora": InputSpec(224, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, False),
    "clip_probe": InputSpec(224, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, False),
    "clip_zeroshot": InputSpec(224, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, False),
    "effnet_b4": InputSpec(256, IMAGENET_MEAN, IMAGENET_STD, True),
}
