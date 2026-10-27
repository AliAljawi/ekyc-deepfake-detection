"""
build_index.py - step 1: the master list of every face image the project uses.

Writes:
  outputs/index/frames.csv.gz           one row per frame (307,413 rows). Gitignored: it holds labels derived from
                                        the datasets and from Xu et al.'s annotations, which aren't redistributed
  outputs/index/frames.sha256           fingerprint of the list's content (stamped into every later run)
  outputs/results/index_summary.csv     counts only (videos, frames, clusters, gender labels per set and method)
  outputs/results/data_provenance.csv   with --provenance: size and SHA-256 of every dataset zip and metadata file

Columns: path (relative to datasets\\, forward slashes), set, source, method, attack_type, label (0 real, 1 fake),
video (set/method/folder, unique), cluster (bootstrap unit = the source clip whose pixels the fake shows: FF++
official pair; Celeb-DF/DF40 target clip cdf/idX_NNNN, except FOMM, which animates clip idY_NNNN; a single image for
the diffusion methods), identity, frame_idx, gender (M/F/U/NA), gender_src, ethnicity, age (FF++ only), src_res.

Sources: FF++ from DeepfakeBench's FaceForensics++.json; Celeb-DF from DeepfakeBench's Celeb-DF-v2.json, test split
only (its "val" is a copy of test and its "train" holds every test fake), checked against the official
List_of_testing_videos.txt; DF40 by listing its folders. Every count is checked against the numbers verified on
Oct 21-22, and every image is opened once so a corrupt file can't crash a training run.

Usage (VS Code terminal, inside the code folder):
    .venv\\Scripts\\python.exe -B build_index.py                 # build, check, open every image once (~3 min)
    .venv\\Scripts\\python.exe -B build_index.py --no-decode     # skip opening every image
    .venv\\Scripts\\python.exe -B build_index.py --provenance    # also fingerprint the dataset zips (~1 min more)
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import (ATTACK_TYPE, DATA, DF40_METHODS, INDEX, LOGS, METHODS, RESULTS, ROOT,  # noqa: E402
                         SETS, ZIPS)
from ekyc.utils import atomic_write, setup_console, setup_logging  # noqa: E402

FFPP_CLASSES = {"FF-real": ("real", "youtube"), "FF-DF": ("DF", "Deepfakes"), "FF-F2F": ("F2F", "Face2Face"),
                "FF-FS": ("FS", "FaceSwap"), "FF-NT": ("NT", "NeuralTextures")}
DIFFUSION_DIRS = {"sd2.1": ("Celeb-real", "YouTube-real"), "DiT": ("Fake_from_Celeb-real", "Fake_from_Youtube-real")}

# Counts verified on Oct 21-22 (videos, frames)
EXPECTED_FFPP = {
    ("real", "train"): (719, 22_993), ("real", "val"): (140, 4_477), ("real", "test"): (140, 4_479),
    ("DF", "train"): (719, 22_928), ("DF", "val"): (140, 4_442), ("DF", "test"): (140, 4_473),
    ("F2F", "train"): (719, 22_991), ("F2F", "val"): (140, 4_478), ("F2F", "test"): (140, 4_480),
    ("FS", "train"): (720, 22_999), ("FS", "val"): (140, 4_478), ("FS", "test"): (140, 4_477),
    ("NT", "train"): (719, 22_973), ("NT", "val"): (140, 4_479), ("NT", "test"): (140, 4_479),
}
EXPECTED_CDF = {"Celeb-real": (108, 3_410), "YouTube-real": (70, 2_210), "Celeb-synthesis": (340, 10_800)}
EXPECTED_DF40 = {"inswap": (424, 13_433), "simswap": (641, 20_322), "fomm": (647, 20_278),
                 "sadtalker": (660, 20_986), "sd2.1": (888, 28_174), "DiT": (888, 28_174)}
EXPECTED_TARGET_CLIPS = {"inswap": 267, "simswap": 320, "fomm": 323, "sadtalker": 325}  # distinct idX_NNNN
EXPECTED_ROWS = 307_413
EXPECTED_GENDER = {  # videos by apparent gender (M, F, U)
    ("cdf_test", "real"): (101, 74, 3), ("cdf_test", "Celeb-synthesis"): (275, 61, 4),
    ("ffpp_test", "real"): (60, 78, 2), ("ffpp_val", "real"): (72, 67, 1),
}

FF_ANN = re.compile(r"^FaceForensics\+\+/(?:original_sequences/(youtube)|manipulated_sequences/"
                    r"(Deepfakes|Face2Face|FaceSwap|NeuralTextures|FaceShifter))/raw/face_images/"
                    r"(\d{3}(?:_\d{3})?)/frame\d+\.png$")
CDF_ANN = re.compile(r"^Celeb-DF-v2_faces_single/(Celeb-real|Celeb-synthesis|YouTube-real)/([^/]+)/frame\d+\.png$")
CELEB_REAL = re.compile(r"^(id\d+)_(\d{4})$")
SWAP = re.compile(r"^(id\d+)_(id\d+)_(\d{4})$")                       # target first: idX_idY_NNNN
SADTALKER = re.compile(r"^(id\d+)_(.{11})_(\d{5})_(\d{4})$")          # idX_<YouTube id>_<clip>_NNNN


def expect(ok, msg):
    if not ok:
        raise AssertionError(msg)


def frame_idx(name):
    """Frame number from a file name: 000.png -> 0, 1021.png -> 1021, Celeb-real_id13_0011_045.png -> 45."""
    return int(Path(name).stem.rsplit("_", 1)[-1])


def row(path, set_, source, method, folder, cluster, identity):
    return {"path": path, "set": set_, "source": source, "method": method, "folder": folder,
            "cluster": cluster, "identity": identity, "frame_idx": frame_idx(path)}


def official_pairs():
    """FF++ id -> (split, sorted official pair). Each id is in exactly one pair, so splits can't share people."""
    pair_of, split_of = {}, {}
    for split in ("train", "val", "test"):
        for a, b in json.loads((DATA / "FaceForensics++" / f"{split}.json").read_text(encoding="utf-8")):
            pair = "_".join(sorted((a, b)))
            for i in (a, b):
                expect(i not in pair_of, f"FF++ id {i} is in two official pairs")
                pair_of[i], split_of[i] = pair, split
    expect(len(pair_of) == 1000, f"FF++ official splits list {len(pair_of)} ids, expected 1000")
    return pair_of, split_of


def ffpp_rows(log):
    meta = json.loads((DATA / "DeepfakeBench_json" / "FaceForensics++.json").read_text(encoding="utf-8"))
    meta = meta["FaceForensics++"]
    pair_of, split_of = official_pairs()
    rows = []
    for cls, (method, source) in FFPP_CLASSES.items():
        for split in ("train", "val", "test"):
            videos = meta[cls][split]["c23"]
            n_frames = 0
            for key, entry in videos.items():
                expect(entry["label"] == cls, f"FF++ {key}: label {entry['label']!r}, expected {cls!r}")
                ids = key.split("_")
                expect(len(ids) == (1 if method == "real" else 2), f"FF++ {cls} key {key}: unexpected form")
                expect(all(split_of[i] == split for i in ids), f"FF++ {key} is listed in {split} but its ids aren't")
                pair = pair_of[ids[0]]
                expect(all(pair_of[i] == pair for i in ids), f"FF++ fake {key} mixes two official pairs")
                for f in entry["frames"]:
                    rel = f.replace("\\", "/")
                    parts = rel.split("/")
                    expect(parts[0] == "FaceForensics++" and parts[2] == source and parts[3:5] == ["c23", "frames"]
                           and parts[5] == key, f"FF++ frame path doesn't match its video: {rel}")
                    rows.append(row(rel, f"ffpp_{split}", source, method, key, f"ffpp/{pair}", f"ffpp/{pair}"))
                n_frames += len(entry["frames"])
            expect((len(videos), n_frames) == EXPECTED_FFPP[method, split],
                   f"FF++ {method} {split}: {len(videos)} videos / {n_frames} frames, "
                   f"expected {EXPECTED_FFPP[method, split]}")
    log.info(f"[ OK ] FF++: {len(rows):,} frames; video and frame counts match for all 5 classes x 3 splits; "
             "every video belongs to an official pair of its own split (no person in two splits)")
    return rows


def cdf_rows(log):
    meta = json.loads((DATA / "DeepfakeBench_json" / "Celeb-DF-v2.json").read_text(encoding="utf-8"))["Celeb-DF-v2"]
    official = {}
    for line in (DATA / "Celeb-DF-v2" / "List_of_testing_videos.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            flag, name = line.split()
            sub, file = name.split("/")
            official[sub, file.removesuffix(".mp4")] = 1 - int(flag)  # that list uses 1 = real
    rows, seen = [], {}
    for cls, label in (("CelebDFv2_real", 0), ("CelebDFv2_fake", 1)):
        for key, entry in meta[cls]["test"].items():  # test only: never "train" or "val" (see the docstring)
            expect(entry["label"] == cls, f"Celeb-DF {key}: label {entry['label']!r}, expected {cls!r}")
            frames = [f.replace("\\", "/") for f in entry["frames"]]
            sub = frames[0].split("/")[1]
            seen[sub, key] = label
            if sub == "Celeb-real":
                m = CELEB_REAL.match(key)
                expect(m is not None, f"unexpected Celeb-real name {key}")
                cluster, identity = f"cdf/{key}", f"cdf/{m[1]}"
            elif sub == "YouTube-real":
                expect(re.fullmatch(r"\d{5}", key) is not None, f"unexpected YouTube-real name {key}")
                cluster = identity = f"cdf/yt_{key}"
            else:
                m = SWAP.match(key)
                expect(m is not None, f"unexpected Celeb-synthesis name {key}")
                cluster, identity = f"cdf/{m[1]}_{m[3]}", f"cdf/{m[1]}"
            method = "real" if label == 0 else "Celeb-synthesis"
            expect((sub == "Celeb-synthesis") == (label == 1), f"Celeb-DF {sub}/{key} has label {label}")
            for rel in frames:
                parts = rel.split("/")
                expect(parts[0] == "Celeb-DF-v2" and parts[1] == sub and parts[2] == "frames" and parts[3] == key,
                       f"Celeb-DF frame path doesn't match its video: {rel}")
                rows.append(row(rel, "cdf_test", sub, method, key, cluster, identity))
    expect(seen == official, "Celeb-DF test split differs from List_of_testing_videos.txt (videos or labels)")
    for sub, expected in EXPECTED_CDF.items():
        n = (sum(1 for s, _ in seen if s == sub), sum(1 for r in rows if r["source"] == sub))
        expect(n == expected, f"Celeb-DF {sub}: {n[0]} videos / {n[1]} frames, expected {expected}")
    log.info(f"[ OK ] Celeb-DF test: {len(seen)} videos / {len(rows):,} frames, identical to the official test list")
    return rows


def df40_rows(log):
    rows = []
    for m in DF40_METHODS:
        base = DATA / "DF40" / m / "cdf"
        subdirs = DIFFUSION_DIRS.get(m, ("frames",))
        expected_top = set(subdirs) if m in DIFFUSION_DIRS else {"frames", "landmarks"}
        top = {e.name for e in os.scandir(base)}
        expect(top == expected_top, f"DF40 {m}/cdf holds {sorted(top)}, expected {sorted(expected_top)}")
        n_folders = n_frames = 0
        for sub in subdirs:
            for d in sorted(os.scandir(base / sub), key=lambda e: e.name):
                expect(d.is_dir(), f"DF40 {m}/cdf/{sub}: unexpected file {d.name}")
                names = [e.name for e in os.scandir(d.path)]
                expect(names and all(n.endswith(".png") for n in names), f"DF40 {m}/{sub}/{d.name}: non-PNG files")
                names.sort(key=lambda n: (frame_idx(n), n))
                source = "cdf" if sub == "frames" else sub
                folder = d.name if sub == "frames" else f"{sub}/{d.name}"
                for n in names:
                    rel = f"DF40/{m}/cdf/{sub}/{d.name}/{n}"
                    if m in DIFFUSION_DIRS:  # random batches of unrelated images: each image is its own unit
                        cluster = identity = rel
                    else:
                        g = (SADTALKER if m == "sadtalker" else SWAP).match(d.name)
                        expect(g is not None, f"DF40 {m}: unexpected folder name {d.name}")
                        # The unit is the clip whose pixels the fake shows: the first id for the swaps and SadTalker,
                        # but FOMM animates a still of clip idY_NNNN with idX's motion (pixel check, Oct 22: 11/11)
                        src = g[2] if m == "fomm" else g[1]
                        cluster, identity = f"cdf/{src}_{g[g.re.groups]}", f"cdf/{src}"
                    rows.append(row(rel, "df40_test", source, m, folder, cluster, identity))
                n_folders += 1
                n_frames += len(names)
        expect((n_folders, n_frames) == EXPECTED_DF40[m],
               f"DF40 {m}: {n_folders} folders / {n_frames} frames, expected {EXPECTED_DF40[m]}")
    log.info(f"[ OK ] DF40: {len(rows):,} frames in 6 methods; folder and frame counts match")
    return rows


def video_attributes(log):
    """Xu et al. (2024) labels per video. They're per frame from the authors' own frame extraction, so they join on
    the video and take the majority (each label is in fact constant within every video)."""
    import pandas as pd

    tables = []
    for file, pattern, cols in (("A-FF++.csv", FF_ANN, ["male", "white", "black", "asian", "young", "middle_aged",
                                                        "senior"]),
                                ("A-Celeb-DF.csv", CDF_ANN, ["male", "white", "black"])):
        ann = pd.read_csv(DATA / "annotations" / file, usecols=["path", "label", *cols])
        parts = ann["path"].str.extract(pattern)
        expect(parts.iloc[:, -1].notna().all(),
               f"{file}: {int(parts.iloc[:, -1].isna().sum())} paths don't match the expected pattern")
        folder = parts[0] if parts.shape[1] == 2 else parts[0].fillna(parts[1])  # FF++: youtube or the method
        ann["key"] = folder + "/" + parts.iloc[:, -1]
        g = ann.groupby("key")
        vid = g[["label", *cols]].first()
        mixed = (g[["label", *cols]].nunique() > 1).any(axis=1)
        for key in mixed[mixed].index:  # majority per column; a tie counts as undefined (0)
            sub = ann.loc[ann["key"] == key, ["label", *cols]]
            for c in ["label", *cols]:
                counts = sub[c].value_counts()
                top = counts.index[counts == counts.max()]
                vid.loc[key, c] = top[0] if len(top) == 1 else 0
        log.info(f"       {file}: {len(ann):,} rows, {len(vid):,} videos, {int(mixed.sum())} with labels that vary "
                 "within the video (majority used)")
        tables.append(vid)
    vid = pd.concat(tables)
    vid["gender"] = vid["male"].map({1: "M", -1: "F", 0: "U"})

    def one_hot(r, names):
        on = [n for n in names if r.get(n) == 1]
        return on[0] if len(on) == 1 else "U"

    vid["ethnicity"] = vid.apply(one_hot, axis=1, names=("white", "black", "asian"))
    vid["age"] = vid.apply(one_hot, axis=1, names=("young", "middle_aged", "senior"))
    vid.loc[vid["young"].isna(), "age"] = "NA"  # Celeb-DF has no age columns
    return vid


def identity_genders(vid, log):
    """Apparent gender of each Celeb-DF person (idX), from their annotated Celeb-real videos. DF40 swap/reenactment
    fakes inherit it only where target and source agree (sadtalker: the target, the only person in it)."""
    real = vid[vid.index.str.startswith("Celeb-real/")]
    person = real.index.str.split("/").str[1].str.split("_").str[0]
    out, conflicts = {}, []
    for pid, genders in real.groupby(person)["gender"]:
        defined = set(genders) - {"U"}
        out[pid] = defined.pop() if len(defined) == 1 else "U"
        if len(defined) > 1:
            conflicts.append(pid)
    log.info(f"       Celeb-DF people with an apparent gender: {sum(g != 'U' for g in out.values())} of {len(out)} "
             f"({sum(g == 'M' for g in out.values())} M, {sum(g == 'F' for g in out.values())} F); "
             f"{len(conflicts)} with conflicting videos")
    return out


def add_attributes(df, log):
    vid = video_attributes(log)
    person = identity_genders(vid, log)
    key = df["source"].astype(str) + "/" + df["folder"]
    in_ann = df["set"].isin(["ffpp_train", "ffpp_val", "ffpp_test", "cdf_test"])
    joined = vid.reindex(key[in_ann])
    has = joined["gender"].notna().to_numpy()
    expect((joined["label"].to_numpy()[has] == df.loc[in_ann, "label"].to_numpy()[has]).all(),
           "annotation labels disagree with the dataset labels for some videos")
    for col in ("gender", "ethnicity", "age"):
        df[col] = "NA"
        df.loc[in_ann, col] = joined[col].fillna("NA").to_numpy()
    df["gender_src"] = "none"
    df.loc[in_ann & (df["gender"] != "NA"), "gender_src"] = "video"

    folder = df["folder"]
    for m, rule in (("inswap", "identities"), ("simswap", "identities"), ("fomm", "identities"),
                    ("sadtalker", "target")):
        sel = df["method"] == m
        tok = folder[sel].str.split("_")
        gx = tok.str[0].map(person).fillna("U")
        if rule == "identities":
            gy = tok.str[1].map(person).fillna("U")
            g = gx.where((gx == gy) & gx.isin(["M", "F"]), "U")
        else:
            g = gx
        df.loc[sel, "gender"] = g.to_numpy()
        df.loc[sel, "gender_src"] = rule
    return df


def build(log):
    import pandas as pd
    df = pd.DataFrame(ffpp_rows(log) + cdf_rows(log) + df40_rows(log))
    df["attack_type"] = df["method"].map(ATTACK_TYPE)
    df["label"] = (df["method"] != "real").astype("int8")
    df["video"] = df["set"] + "/" + df["method"] + "/" + df["folder"]
    df["src_res"] = (df["method"] == "sd2.1").map({True: 512, False: 256}).astype("int16")
    df = add_attributes(df, log)
    df["set"] = pd.Categorical(df["set"], categories=SETS, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
    df = df.sort_values(["set", "method", "video", "frame_idx", "path"], kind="stable").reset_index(drop=True)
    cols = ["path", "set", "source", "method", "attack_type", "label", "video", "cluster", "identity", "frame_idx",
            "gender", "gender_src", "ethnicity", "age", "src_res"]
    return df[cols]


def check(df, log):
    expect(len(df) == EXPECTED_ROWS, f"{len(df):,} frames, expected {EXPECTED_ROWS:,}")
    expect(df["path"].is_unique, "a frame is listed twice")
    parent = df["path"].str.rsplit("/", n=1).str[0]
    expect(df.groupby("video")["label"].nunique().eq(1).all(), "a video mixes real and fake frames")
    expect(parent.groupby(df["video"]).nunique().eq(1).all() and parent.nunique() == df["video"].nunique(),
           "video ids don't map one-to-one to folders")
    expect(df.isna().sum().sum() == 0, "missing values in the index")

    test = df[df["set"] == "ffpp_test"]
    per = test.groupby("cluster")["video"].nunique()
    real_per = test[test["label"] == 0].groupby("cluster")["video"].nunique()
    expect(len(per) == 70 and per.eq(10).all() and real_per.eq(2).all(),
           "FF++ test should be 70 official pairs, each with 2 real + 8 fake videos")
    ff_people = {s: set(df.loc[df["set"] == s, "identity"]) for s in ("ffpp_train", "ffpp_val", "ffpp_test")}
    expect(not (ff_people["ffpp_train"] & ff_people["ffpp_val"]) and not (ff_people["ffpp_train"] & ff_people["ffpp_test"])
           and not (ff_people["ffpp_val"] & ff_people["ffpp_test"]), "an FF++ pair appears in two splits")

    df40 = df[df["set"] == "df40_test"]
    for m, n in EXPECTED_TARGET_CLIPS.items():
        sub = df40[df40["method"] == m]
        expect(sub["cluster"].nunique() == n and sub["identity"].nunique() == 55,
               f"DF40 {m}: {sub['cluster'].nunique()} target clips / {sub['identity'].nunique()} people, "
               f"expected {n} / 55")
    for m in ("sd2.1", "DiT"):
        sub = df40[df40["method"] == m]
        expect(sub["cluster"].is_unique, f"DF40 {m}: each image should be its own cluster")

    vids = df.drop_duplicates("video")
    for (s, m), expected in EXPECTED_GENDER.items():
        g = vids.loc[(vids["set"] == s) & (vids["method"] == m), "gender"].value_counts()
        got = (int(g.get("M", 0)), int(g.get("F", 0)), int(g.get("U", 0)))
        expect(got == expected and int(g.get("NA", 0)) == 0,
               f"{s} {m}: apparent gender M/F/U = {got} (NA {int(g.get('NA', 0))}), expected {expected}")
    log.info(f"[ OK ] {len(df):,} frames; checks passed: unique paths and videos, FF++ test = 70 pairs x (2 real + "
             "8 fake), no FF++ pair in two splits, DF40 target clips and people, gender counts")


def summary(df):
    import pandas as pd
    g = df.groupby(["set", "method"], observed=True)
    out = g.agg(videos=("video", "nunique"), frames=("path", "size"), clusters=("cluster", "nunique"),
                identities=("identity", "nunique"))
    vids = df.drop_duplicates("video")
    gender = pd.crosstab([vids["set"], vids["method"]], vids["gender"]).reindex(columns=["M", "F", "U", "NA"],
                                                                                   fill_value=0)
    out = out.join(gender.add_prefix("videos_gender_"))
    out.insert(0, "attack_type", [ATTACK_TYPE[m] for _, m in out.index])
    return out.reset_index()


def _decode_chunk(job):
    """Runs in a worker process: opens every image in the chunk and checks it's RGB at the expected size."""
    from torchvision.io import ImageReadMode, decode_image, read_file
    root, rels, sizes = job
    bad, nbytes = [], 0
    for rel, size in zip(rels, sizes):
        try:
            data = read_file(os.path.join(root, rel))
            nbytes += data.numel()
            img = decode_image(data, mode=ImageReadMode.UNCHANGED)
            if tuple(img.shape) != (3, size, size):
                bad.append(f"{rel}: decoded to {tuple(img.shape)}, expected (3, {size}, {size})")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{rel}: {type(e).__name__}: {e}")
    return len(rels), nbytes, bad


def decode_pass(df, log, workers):
    from concurrent.futures import ProcessPoolExecutor
    paths, sizes = df["path"].tolist(), df["src_res"].tolist()
    step = 2_000
    jobs = [(str(DATA), paths[i:i + step], sizes[i:i + step]) for i in range(0, len(paths), step)]
    log.info(f"Opening all {len(paths):,} images once ({workers} processes)...")
    t0, done, nbytes, bad, next_report = time.time(), 0, 0, [], 0.25
    with ProcessPoolExecutor(workers) as ex:
        for n, nb, b in ex.map(_decode_chunk, jobs):
            done, nbytes, bad = done + n, nbytes + nb, bad + b
            if done / len(paths) >= next_report:
                log.info(f"       {done / len(paths):4.0%}  {done / (time.time() - t0):,.0f} img/s")
                next_report += 0.25
    secs = time.time() - t0
    if bad:
        for b in bad[:20]:
            log.info(f"       {b}")
        raise AssertionError(f"{len(bad)} image(s) failed to open or have the wrong size (first 20 above)")
    log.info(f"[ OK ] all {done:,} images open as RGB at the expected size ({nbytes / 1e9:.1f} GB in {secs:.0f} s, "
             f"{done / secs:,.0f} img/s)")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return path, path.stat().st_size, h.hexdigest()


def provenance(log):
    from concurrent.futures import ThreadPoolExecutor
    files = sorted(ZIPS.glob("*.zip")) + [
        DATA / "DeepfakeBench_json" / "FaceForensics++.json", DATA / "DeepfakeBench_json" / "Celeb-DF-v2.json",
        DATA / "FaceForensics++" / "train.json", DATA / "FaceForensics++" / "val.json",
        DATA / "FaceForensics++" / "test.json", DATA / "Celeb-DF-v2" / "List_of_testing_videos.txt",
        DATA / "annotations" / "A-FF++.csv", DATA / "annotations" / "A-Celeb-DF.csv"]
    log.info(f"Fingerprinting {len(files)} dataset files ({sum(f.stat().st_size for f in files) / 1e9:.1f} GB)...")
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(_sha256, files))
    lines = ["file,bytes,sha256"] + [f"{p.relative_to(ROOT).as_posix()},{n},{h}" for p, n, h in results]
    atomic_write(RESULTS / "data_provenance.csv", "\n".join(lines) + "\n")
    log.info(f"[ OK ] SHA-256 of {len(files)} files in {time.time() - t0:.0f} s -> outputs/results/data_provenance.csv")


def main():
    ap = argparse.ArgumentParser(description="Build the master list of frames (step 1).")
    ap.add_argument("--no-decode", action="store_true", help="skip opening every image")
    ap.add_argument("--provenance", action="store_true", help="also fingerprint the dataset zips")
    ap.add_argument("--workers", type=int, default=8, help="processes for the decode pass")
    args = ap.parse_args()
    setup_console()
    log = setup_logging(LOGS / "build_index.log")
    t0 = time.time()

    df = build(log)
    check(df, log)
    csv = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    sha = hashlib.sha256(csv).hexdigest()
    atomic_write(INDEX, gzip.compress(csv, compresslevel=6, mtime=0))
    atomic_write(INDEX.with_name("frames.sha256"), sha + "\n")
    atomic_write(RESULTS / "index_summary.csv", summary(df).to_csv(index=False, lineterminator="\n"))
    log.info(f"[ OK ] wrote {INDEX.relative_to(ROOT)} ({INDEX.stat().st_size / 1e6:.1f} MB, sha256 {sha[:12]}...) "
             "and outputs/results/index_summary.csv")

    if not args.no_decode:
        decode_pass(df, log, args.workers)
    if args.provenance:
        provenance(log)
    log.info(f"Done in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
