"""
Exports everything the in-class demo needs to outputs/demo/ (gitignored: it holds weights and dataset faces).
Copy that folder to a private Google Drive folder named ekyc_demo, then run notebooks/demo_gradio.ipynb in Colab.

What it writes:
  vit.safetensors         CLIP ViT-B/16 vision tower with the seed-0 LoRA merged in (plain weights, no PEFT needed)
                          plus the linear head
  effnet_b4.safetensors   the seed-0 EfficientNet-B4 (timm efficientnet_b4, 1 output)
  config.json             model settings, input sizes and normalisation, the fixed 1% thresholds, the faces
  faces/A.png ... H.png   8 prepared 256x256 test faces: the 4 spot-the-fake slide faces (A-D), the 2 faces from
                          the compression figure (E, F), a reenactment the ViT catches (G) and a genuine face both
                          models wrongly reject (H); all taken from outputs/samples/showcase_manifest.json
  demo_app.py, demo_gradio.ipynb, README.txt   the app, the Colab notebook and how to run them
  check.json              the checks below

Checks (fp32 on the CPU, the same path the demo uses):
  1. the merged ViT matches the LoRA model: max |logit difference| < 1e-3 on every face and JPEG level checked
  2. every face gets the same decision as the frozen scores in the showcase manifest (they were scored on the GPU
     under bf16, so logits differ slightly; decisions must not)
  3. PIL's JPEG round trip (used by the demo) reproduces ekyc.data.jpeg (used in the evaluation)

usage:  .venv\\Scripts\\python.exe export_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ekyc.config  # noqa: E402,F401  (first: sets the Hugging Face offline variables)

RUNS = {"vit": "clip_lora_s0", "effnet": "effnet_b4_s0"}
NAMES = {"vit": "ViT (CLIP + LoRA)", "effnet": "EfficientNet-B4"}  # the names used in the manifest: ViT, EfficientNet
MANIFEST_KEY = {"vit": "ViT", "effnet": "EfficientNet"}
QUALITY = {"png": None, "q90": 90, "q75": 75, "q60": 60, "q45": 45, "q30": 30}
MERGE_TOL = 1e-3


def pick_faces(manifest):
    """The 8 demo faces, in display order, from the showcase manifest."""
    fig = manifest["figures"]
    chosen = [("quiz", e) for e in fig["spot_the_fake"]]
    chosen += [("compression", e) for e in fig["compression"]]
    chosen.append(("caught", next(e for e in fig["heatmaps"] if e["group"] == "reenactment, caught by the ViT")))
    chosen.append(("rejected", fig["rejected_genuine"][0]))
    faces = []
    for i, (role, e) in enumerate(chosen):
        key = e.get("letter") or "ABCDEFGH"[i]
        faces.append({"key": key, "file": f"{key}.png", "role": role, "row": int(e["row"]), "path": e["path"],
                      "truth": e["shown_as"], "fake": not e["shown_as"].startswith("Genuine"),
                      "reference": {m: e[MANIFEST_KEY[m]] for m in MANIFEST_KEY}})
    keys = [f["key"] for f in faces]
    if keys != list("ABCDEFGH"):
        raise AssertionError(f"unexpected face order {keys}")
    return faces


def main():
    import io
    import json
    import shutil
    import subprocess
    import time

    import numpy as np
    import pandas as pd
    import peft
    import timm
    import torch
    import transformers
    from PIL import Image
    from safetensors.torch import save

    from ekyc.config import DATA, OUT, RESULTS, SAMPLES, SPEC
    from ekyc.data import decode_256, jpeg
    from ekyc.models import load_checkpoint
    from ekyc.utils import atomic_write

    code = Path(__file__).resolve().parent
    demo = OUT / "demo"
    (demo / "faces").mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    manifest = json.loads((SAMPLES / "showcase_manifest.json").read_text(encoding="utf-8"))
    faces = pick_faces(manifest)
    thr_table = pd.read_csv(RESULTS / "thresholds.csv")
    thr = {}
    for m, run in RUNS.items():
        sel = thr_table[(thr_table["run"] == run) & np.isclose(thr_table["target_bpcer"], 0.01)]
        if len(sel) != 1:
            raise AssertionError(f"thresholds.csv: expected one 1% row for {run}, found {len(sel)}")
        thr[m] = float(sel["threshold"].iloc[0])
    for m in RUNS:  # the manifest was made with the same thresholds
        if abs(thr[m] - manifest["thresholds"][MANIFEST_KEY[m]]) > 1e-4:
            raise AssertionError(f"{m}: threshold {thr[m]} differs from the manifest's")

    # faces: decoded exactly as in the evaluation (SD-2.1's 512x512 images resized to 256 first)
    imgs = {}
    for f in faces:
        src = DATA / f["path"]
        with Image.open(src) as im:
            is512 = im.size == (512, 512)
        img = decode_256(str(src), is512)
        imgs[f["key"]] = img
        Image.fromarray(img.permute(1, 2, 0).contiguous().numpy()).save(demo / "faces" / f["file"])

    # every (face, level) the manifest has a frozen score for
    cases = [(f["key"], lv) for f in faces for lv in f["reference"]["vit"]]
    batch = torch.stack([imgs[k] if QUALITY[lv] is None else jpeg(imgs[k], QUALITY[lv]) for k, lv in cases])

    vit, vit_meta = load_checkpoint(OUT / "checkpoints" / RUNS["vit"] / "best.safetensors", device="cpu")
    eff, eff_meta = load_checkpoint(OUT / "checkpoints" / RUNS["effnet"] / "best.safetensors", device="cpu")
    lora_logits = vit(batch)
    eff_logits = eff(batch)
    vit.backbone = vit.backbone.merge_and_unload()  # LoRA folded into the attention weights: a plain CLIPVisionModel
    merged_logits = vit(batch)
    merge_diff = float((merged_logits - lora_logits).abs().max())
    if merge_diff >= MERGE_TOL:
        raise AssertionError(f"merged ViT differs from the LoRA model by {merge_diff:.2e} (limit {MERGE_TOL})")

    # the exported weights
    vit_sd = {f"backbone.{k}": v for k, v in vit.backbone.state_dict().items()}
    vit_sd.update({f"head.{k}": v for k, v in vit.head.state_dict().items()})
    eff_sd = {k.removeprefix("net."): v for k, v in eff.state_dict().items()}
    atomic_write(demo / "vit.safetensors", save({k: v.detach().float().contiguous() for k, v in vit_sd.items()}))
    atomic_write(demo / "effnet_b4.safetensors", save({k: v.detach().contiguous() for k, v in eff_sd.items()}))

    # decisions against the frozen scores
    rows, mismatches = [], 0
    for i, (k, lv) in enumerate(cases):
        f = next(x for x in faces if x["key"] == k)
        for m, logit in (("vit", float(merged_logits[i])), ("effnet", float(eff_logits[i]))):
            ref = f["reference"][m][lv]
            called = "fake" if logit > thr[m] else "genuine"
            ok = called == ref["called"]
            mismatches += not ok
            rows.append({"face": k, "level": lv, "model": m, "logit": round(logit, 4),
                         "frozen_logit": ref["logit"], "diff": round(logit - ref["logit"], 4),
                         "called": called, "frozen_called": ref["called"], "same_decision": ok})

    # the demo compresses with PIL; the evaluation used torchvision's libjpeg-turbo round trip
    def pil_jpeg(arr, q):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"))
    jpeg_px = {}
    for k, img in imgs.items():
        arr = img.permute(1, 2, 0).contiguous().numpy()
        for q in (90, 60, 30):
            jpeg_px[f"{k}_q{q}"] = int((pil_jpeg(arr, q) != jpeg(img, q).permute(1, 2, 0).numpy()).sum())

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=code, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    config = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "git_commit": commit,
        "vit": {"name": NAMES["vit"], "file": "vit.safetensors", "run": RUNS["vit"], "epoch": vit_meta.get("epoch"),
                "clip_vision_config": vit.backbone.config.to_dict(), "size": SPEC["clip_lora"].size,
                "mean": list(SPEC["clip_lora"].mean), "std": list(SPEC["clip_lora"].std), "threshold": thr["vit"]},
        "effnet": {"name": NAMES["effnet"], "file": "effnet_b4.safetensors", "run": RUNS["effnet"],
                   "epoch": eff_meta.get("epoch"), "timm_name": "efficientnet_b4", "size": SPEC["effnet_b4"].size,
                   "mean": list(SPEC["effnet_b4"].mean), "std": list(SPEC["effnet_b4"].std),
                   "threshold": thr["effnet"]},
        "threshold_rule": "1% of genuine FF++ validation frames rejected (seed 0), as in the report",
        "faces": [{k: f[k] for k in ("key", "file", "role", "truth", "fake", "path")} for f in faces],
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "timm": timm.__version__,
                     "peft": peft.__version__},
        "credit": "Faces: Celeb-DF v2 (Li et al., 2020) and DF40 (Yan et al., 2024; CC BY-NC 4.0), research "
                  "datasets, resized; shown for this assessment only, not for redistribution.",
    }
    atomic_write(demo / "config.json", json.dumps(config, indent=1))
    check = {"merge_max_abs_diff": merge_diff, "merge_tolerance": MERGE_TOL,
             "decision_mismatches": mismatches, "max_abs_diff_vs_frozen": max(abs(r["diff"]) for r in rows),
             "jpeg_pixels_differing_pil_vs_eval": jpeg_px, "cases": rows}
    atomic_write(demo / "check.json", json.dumps(check, indent=1))
    shutil.copy2(code / "demo_app.py", demo / "demo_app.py")
    shutil.copy2(code / "notebooks" / "demo_README.txt", demo / "README.txt")
    shutil.copy2(code / "notebooks" / "demo_gradio.ipynb", demo / "demo_gradio.ipynb")

    size = sum(p.stat().st_size for p in demo.rglob("*") if p.is_file()) / 1e6
    print(f"Wrote {demo} ({size:.0f} MB): vit.safetensors, effnet_b4.safetensors, config.json, 8 faces")
    print(f"1. merged ViT vs LoRA model: max |logit diff| {merge_diff:.1e} (limit {MERGE_TOL:.0e})  OK")
    print(f"2. decisions vs frozen scores: {len(rows) - mismatches}/{len(rows)} identical; "
          f"max |logit diff| {check['max_abs_diff_vs_frozen']:.3f}")
    worst = max(jpeg_px.values())
    print(f"3. PIL JPEG vs evaluation JPEG: {'identical' if worst == 0 else f'up to {worst} pixel values differ'}")
    for f in faces:
        rs = [r for r in rows if r["face"] == f["key"] and r["level"] == "png"]
        calls = ", ".join(f"{NAMES[r['model']]}: {r['called']}" for r in rs)
        print(f"   {f['key']}  {f['truth']:<36} {calls}")
    if mismatches:
        print(f"WARNING: {mismatches} decision(s) differ from the frozen scores; see check.json before using the demo")
        sys.exit(1)
    print("Next: copy outputs/demo to Google Drive as MyDrive/ekyc_demo and open notebooks/demo_gradio.ipynb in Colab.")


if __name__ == "__main__":
    main()
