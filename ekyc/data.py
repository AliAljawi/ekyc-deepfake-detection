"""
Reading frames: the master list, image decoding, JPEG copies, training augmentation, the balanced epoch sampler
and DataLoaders.

DataLoader workers on Windows start fresh ("spawn") and unpickle the dataset, so everything they need lives in this
module, which imports only numpy/torch/torchvision at the top (pandas is imported inside the index functions).
"""
import gzip
import hashlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import ImageReadMode, decode_image, decode_jpeg, encode_jpeg
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from ekyc.config import ATTACK_TYPES, CACHE, DATA, INDEX, K_FAKE, K_REAL, METHODS, SETS

CATEGORIES = ("set", "source", "method", "attack_type", "gender", "gender_src", "ethnicity", "age")
ORDER = {"set": SETS, "method": METHODS, "attack_type": ("real", *ATTACK_TYPES), "gender": ("M", "F", "U", "NA")}


def load_index(columns=None):
    """The master list (one row per frame); the row number is the frame's id in every later file."""
    import pandas as pd
    if not INDEX.is_file():
        raise FileNotFoundError(f"{INDEX} not found: run build_index.py first")
    wanted = CATEGORIES if columns is None else [c for c in CATEGORIES if c in columns]
    # keep_default_na=False: the gender value "NA" (not annotated) must stay a string, not become a missing value
    df = pd.read_csv(INDEX, usecols=columns, keep_default_na=False, dtype={c: "category" for c in wanted})
    for col, order in ORDER.items():
        if col in df:
            df[col] = df[col].cat.set_categories(order, ordered=True)
    return df


def index_sha256():
    """Fingerprint of the master list's content, stamped into every run so results can't be joined to a rebuilt list."""
    h = hashlib.sha256()
    with gzip.open(INDEX, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def decode_256(abs_path, is512=False):
    """uint8 RGB tensor (3, 256, 256). The 512x512 SD-2.1 images are resized first (bicubic, antialiased), so image
    size can't act as a shortcut."""
    img = decode_image(abs_path, mode=ImageReadMode.RGB)
    if is512:
        img = TF.resize(img, [256, 256], interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
    if img.shape != (3, 256, 256):
        raise ValueError(f"{abs_path}: decoded to {tuple(img.shape)}, expected (3, 256, 256)")
    return img


def jpeg(img, quality):
    """JPEG round trip (libjpeg-turbo, 4:2:0, standard tables; byte-identical to PIL)."""
    return decode_jpeg(encode_jpeg(img, quality=quality), mode=ImageReadMode.RGB)


def train_transform():
    """Light augmentation only: mirror, small zoom, mild brightness/contrast. No blur or JPEG, so the compression
    test stays unseen."""
    return v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.95, 1.05), antialias=True),
        v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
    ])


class FrameDataset(Dataset):
    """Frames listed in the index.

    Training: the key is (position, aug_seed) from BalancedEpochSampler, and the item is (uint8 3x256x256, row).
    The augmentation is drawn from aug_seed alone, so every model sees the same random changes, whichever worker
    loads the frame. Evaluation: the key is a position, and the item is (uint8 Lx3x256x256, row) with one copy per
    JPEG level in jpeg_q (None = the original PNG).
    """

    def __init__(self, rel_paths, rows, is512, *, train, jpeg_q=(None,)):
        self.paths = np.char.encode(np.asarray(rel_paths, dtype=str), "utf-8")  # compact bytes, cheap to pickle
        self.rows = np.asarray(rows, dtype=np.int64)
        self.is512 = np.asarray(is512, dtype=bool)
        self.train, self.jpeg_q = train, tuple(jpeg_q)
        self.root = str(DATA)
        self.aug = train_transform() if train else None

    @classmethod
    def from_index(cls, df, **kw):
        return cls(df["path"].to_numpy(dtype=str), df.index.to_numpy(), df["src_res"].to_numpy() == 512, **kw)

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        """Workers receive a short pickle and load the arrays from a cache file themselves: on Windows, a large
        pickle makes the DataLoader start its workers one after another (~1.4 s each) instead of in parallel."""
        state = self.__dict__.copy()
        for k in ("paths", "rows", "is512"):
            del state[k]
        state["_arrays_file"] = self._arrays_file()
        return state

    def __setstate__(self, state):
        path = state.pop("_arrays_file")
        self.__dict__.update(state)
        with np.load(path) as z:
            self.paths, self.rows, self.is512 = z["paths"], z["rows"], z["is512"]

    def _arrays_file(self):
        if getattr(self, "_cached_file", None) is None:
            h = hashlib.sha256()
            for a in (self.paths, self.rows, self.is512):
                h.update(f"{a.dtype.str}{a.shape}".encode())
                h.update(a.tobytes())
            path = CACHE / "datasets" / f"{h.hexdigest()[:20]}.npz"
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
                np.savez(tmp, paths=self.paths, rows=self.rows, is512=self.is512)
                os.replace(tmp, path)
            self._cached_file = str(path)
        return self._cached_file

    def __getitem__(self, key):
        i, aug_seed = key if self.train else (key, None)
        img = decode_256(os.path.join(self.root, self.paths[i].decode("utf-8")), bool(self.is512[i]))
        if self.train:
            # CPU generator only: torch.manual_seed would also queue a CUDA seed and format a stack trace each call
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(aug_seed)
                img = self.aug(img)
            return img, int(self.rows[i])
        return torch.stack([img if q is None else jpeg(img, q) for q in self.jpeg_q]), int(self.rows[i])


class BalancedEpochSampler(Sampler):
    """Each epoch: k random frames per video (all of them if it has fewer), in random order, as (position, aug_seed).

    It runs in the main process and depends only on (seed, epoch), so every model gets the same frames, order and
    augmentation draws. aug_seed comes from SeedSequence([seed, epoch, position in the epoch]).
    """

    def __init__(self, video_code, video_is_fake, k_real=K_REAL, k_fake=K_FAKE, seed=0):
        self.code = np.asarray(video_code, dtype=np.int64)
        n_v = np.bincount(self.code)
        self.starts = np.concatenate([[0], np.cumsum(n_v)[:-1]])
        self.take = np.minimum(np.where(np.asarray(video_is_fake, dtype=bool), k_fake, k_real), n_v)
        self.seed, self.epoch = seed, 0

    @classmethod
    def from_index(cls, df, seed, k_real=K_REAL, k_fake=K_FAKE):
        import pandas as pd
        code, uniq = pd.factorize(df["video"])
        is_fake = np.zeros(len(uniq), dtype=bool)
        is_fake[code] = df["label"].to_numpy() == 1
        return cls(code, is_fake, k_real, k_fake, seed)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def draws(self, epoch):
        """(positions, aug_seeds) for one epoch."""
        rng = np.random.default_rng([self.seed, epoch])
        order = np.lexsort((rng.random(len(self.code)), self.code))  # grouped by video, random order inside
        rank = np.arange(len(order)) - self.starts[self.code[order]]
        keep = order[rank < self.take[self.code[order]]]
        rng.shuffle(keep)
        seeds = np.array([np.random.SeedSequence([self.seed, epoch, pos]).generate_state(1)[0]
                          for pos in range(len(keep))], dtype=np.uint32)
        return keep, seeds

    def __iter__(self):
        keep, seeds = self.draws(self.epoch)
        return iter(zip(keep.tolist(), seeds.tolist()))

    def __len__(self):
        return int(self.take.sum())


def _worker_init(_worker_id):
    torch.set_num_threads(1)  # one thread per worker; the workers already run in parallel


def make_loader(ds, *, batch_size, num_workers, seed, sampler=None, drop_last=False, prefetch_factor=2):
    # No pinned memory: on this Windows PC the extra pinning thread made training 4-8% slower
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers,
                      pin_memory=False, drop_last=drop_last,
                      persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor if num_workers else None,
                      worker_init_fn=_worker_init, generator=torch.Generator().manual_seed(seed))
