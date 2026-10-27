"""
check_setup.py - run once on the desktop after installing the libraries.

Checks, in order:
  1. Python version and that every library imports
  2. PyTorch can see the RTX 5090 and actually compute on it
  3. CLIP ViT-B/16 (+ LoRA) and EfficientNet-B4 download and train for a few steps
  4. Rough GPU speed for training and inference
  5. The datasets folder has everything the experiments need

Usage (VS Code terminal, with .venv active):
    python check_setup.py
The output is also saved to setup_log.txt next to this file.
"""
import importlib
import os
import platform
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent  # C:\Applied Vision Transformer Report
DATA = ROOT / "datasets"
LIBS = ["torch", "torchvision", "transformers", "peft", "timm", "sklearn",
        "pandas", "numpy", "PIL", "matplotlib", "tqdm"]
CU130 = "pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130"
LOG = Path(__file__).resolve().parent / "setup_log.txt"


class Tee:
    """Prints to the screen and saves a copy to setup_log.txt."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def ok(msg):
    print(f"[ OK ] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def note(msg):
    print(f"       {msg}")


def check_python_and_libs():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()} {platform.release()}, "
          f"{os.cpu_count()} CPU threads")
    if not (3, 10) <= (v.major, v.minor) <= (3, 14):
        fail("use Python 3.10-3.14 (3.12 recommended)")
        return False
    missing = []
    for name in LIBS:
        try:
            mod = importlib.import_module(name)
            note(f"{name:13s} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            missing.append(f"{name} ({type(e).__name__}: {e})")
    if missing:
        fail("missing or broken: " + "; ".join(missing))
        return False
    ok("all libraries import")
    return True


def check_gpu():
    import torch
    note(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        fail("PyTorch can't see the GPU. If the torch version ends in '+cpu', reinstall with:")
        note(CU130)
        return False
    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    ok(f"GPU: {props.name}, {props.total_memory / 1024**3:.0f} GB, compute capability {major}.{minor}")
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        fail(f"this PyTorch build has no kernels for {arch} (it has {torch.cuda.get_arch_list()}). Reinstall with:")
        note(CU130)
        return False
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        (a @ a).float().sum().item()
        torch.cuda.synchronize()
        ok("GPU computes correctly (bf16 matrix multiply)")
    except Exception as e:  # noqa: BLE001
        fail(f"GPU test failed: {e}")
        return False
    return True


def bench(model, size, batch, train, steps=15, warmup=5):
    import torch
    model = model.to("cuda", memory_format=torch.channels_last).train(train)
    x = torch.randn(batch, 3, size, size, device="cuda").contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 2, (batch, 1), device="cuda").float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4) if train else None
    loss_fn = torch.nn.BCEWithLogitsLoss()
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        if train:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
    torch.cuda.synchronize()
    return batch * steps / (time.time() - t0), torch.cuda.max_memory_allocated() / 1024**3


def speed_test():
    import torch
    import timm
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPVisionModel

    class ClipDetector(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = torch.nn.Linear(768, 1)

        def forward(self, x):
            return self.head(self.backbone(pixel_values=x).pooler_output)

    print("Loading models (the first run downloads ~0.5 GB, give it a minute)...")
    try:
        clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        total = sum(p.numel() for p in clip.parameters())
        clip = get_peft_model(clip, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1,
                                              target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
        clip_det = ClipDetector(clip)
        trainable = sum(p.numel() for p in clip_det.parameters() if p.requires_grad)
        ok(f"CLIP ViT-B/16 + LoRA ready: {total / 1e6:.1f}M params, {trainable / 1e6:.2f}M trainable "
           f"({100 * trainable / total:.1f}%)")
        effnet = timm.create_model("efficientnet_b4.ra2_in1k", pretrained=True, num_classes=1)
        ok(f"EfficientNet-B4 ready: {sum(p.numel() for p in effnet.parameters()) / 1e6:.1f}M params")
    except Exception as e:  # noqa: BLE001
        fail(f"model download/build failed: {type(e).__name__}: {e}")
        return False

    try:
        print("Speed test (GPU only, synthetic images)...")
        tr_c, m_c = bench(clip_det, 224, 128, train=True)
        inf_c, _ = bench(clip_det, 224, 256, train=False)
        tr_e, m_e = bench(effnet, 256, 64, train=True)
        inf_e, _ = bench(effnet, 256, 256, train=False)
    except Exception as e:  # noqa: BLE001
        fail(f"speed test failed: {type(e).__name__}: {e}")
        return False
    ok(f"CLIP ViT-B/16 + LoRA @224: train {tr_c:,.0f} img/s (peak {m_c:.1f} GB), inference {inf_c:,.0f} img/s")
    ok(f"EfficientNet-B4  @256: train {tr_e:,.0f} img/s (peak {m_e:.1f} GB), inference {inf_e:,.0f} img/s")
    note(f"=> 30,000 training images: CLIP ~{30000 / tr_c / 60:.1f} min, EfficientNet ~{30000 / tr_e / 60:.1f} min "
         "(GPU only; the next script measures image loading)")
    return True


def check_datasets():
    need = {
        "FF++ real (youtube)": DATA / "FaceForensics++/original_sequences/youtube/c23/frames",
        "FF++ Deepfakes": DATA / "FaceForensics++/manipulated_sequences/Deepfakes/c23/frames",
        "FF++ Face2Face": DATA / "FaceForensics++/manipulated_sequences/Face2Face/c23/frames",
        "FF++ FaceSwap": DATA / "FaceForensics++/manipulated_sequences/FaceSwap/c23/frames",
        "FF++ NeuralTextures": DATA / "FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames",
        "FF++ official test split": DATA / "FaceForensics++/test.json",
        "Celeb-DF real (Celeb-real)": DATA / "Celeb-DF-v2/Celeb-real/frames",
        "Celeb-DF real (YouTube-real)": DATA / "Celeb-DF-v2/YouTube-real/frames",
        "Celeb-DF fake": DATA / "Celeb-DF-v2/Celeb-synthesis/frames",
        "Celeb-DF test list": DATA / "Celeb-DF-v2/List_of_testing_videos.txt",
        "DeepfakeBench FF++ splits": DATA / "DeepfakeBench_json/FaceForensics++.json",
        "DeepfakeBench Celeb-DF splits": DATA / "DeepfakeBench_json/Celeb-DF-v2.json",
        "Labels A-FF++": DATA / "annotations/A-FF++.csv",
        "Labels A-Celeb-DF": DATA / "annotations/A-Celeb-DF.csv",
    }
    for m in ["inswap", "simswap", "fomm", "sadtalker", "sd2.1", "DiT"]:
        need[f"DF40 {m}"] = DATA / "DF40" / m / "cdf"
    missing = [f"{k} ({v})" for k, v in need.items() if not v.exists()]
    if missing:
        fail("datasets not found: " + "; ".join(missing))
    else:
        ok(f"all {len(need)} dataset folders/files found under {DATA}")
    note(f"free disk space on this drive: {shutil.disk_usage(ROOT).free / 1e9:,.0f} GB")
    return not missing


def main():
    log = open(LOG, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    print("=" * 70)
    results = {"python+libs": check_python_and_libs()}
    results["gpu"] = results["python+libs"] and check_gpu()
    results["models+speed"] = results["gpu"] and speed_test()
    results["datasets"] = check_datasets()
    print("=" * 70)
    if all(results.values()):
        print("ALL CHECKS PASSED.")
    else:
        print("Some checks failed: " + ", ".join(k for k, v in results.items() if not v))
    print(f"Output saved to {LOG.name}.")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
