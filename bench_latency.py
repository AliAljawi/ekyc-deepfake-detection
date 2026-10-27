"""
bench_latency.py - speed per selfie: how long each model takes to judge one 256x256 face crop, on the GPU (bf16, as
evaluated) and on the CPU (fp32), plus GPU throughput in batches of 64. Timing includes the pre-processing (resize and
normalisation) but not reading the file. Weights don't affect speed, so freshly built models are timed; the ViT is
timed both as evaluated (LoRA kept separate) and merged (as it would be deployed).

Writes outputs/results/latency.csv. Run it when nothing else is using the GPU.

Usage (VS Code terminal, inside the code folder):
    .venv\\Scripts\\python.exe -B bench_latency.py
"""
import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekyc.config import CACHE, RESULTS  # noqa: E402
from ekyc.utils import atomic_write, setup_console  # noqa: E402


def timed(model, x, runs, warmup):
    import torch
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def main():
    ap = argparse.ArgumentParser(description="Per-image latency and throughput of each model.")
    ap.add_argument("--cpu-runs", type=int, default=50)
    args = ap.parse_args()
    setup_console()
    import pandas as pd
    import torch
    from ekyc.models import ZeroShotCLIP, build_model

    torch.backends.cudnn.benchmark = True
    cpu_name = platform.processor() or platform.machine()
    models = {"clip_lora": lambda: build_model("clip_lora"), "clip_probe": lambda: build_model("clip_probe"),
              "effnet_b4": lambda: build_model("effnet_b4"), "clip_zeroshot": lambda: ZeroShotCLIP(CACHE)}
    rows = []
    for kind, make in models.items():
        variants = [("as evaluated", make().eval())]
        if kind == "clip_lora":
            merged = make().eval()
            merged.backbone = merged.backbone.merge_and_unload()
            variants.append(("LoRA merged (deployment)", merged))
        for variant, model in variants:
            gpu = model.cuda()
            x1 = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8, device="cuda")
            x64 = torch.randint(0, 256, (64, 3, 256, 256), dtype=torch.uint8, device="cuda")
            t1 = timed(gpu, x1, 200, 30)
            t64 = timed(gpu, x64, 30, 10)
            rows.append({"model": kind, "variant": variant, "device": torch.cuda.get_device_name(0),
                         "precision": "bf16 autocast", "batch": 1, "median_ms": 1e3 * statistics.median(t1),
                         "p90_ms": 1e3 * sorted(t1)[int(0.9 * len(t1))], "images_per_s": 1 / statistics.median(t1)})
            rows.append({"model": kind, "variant": variant, "device": torch.cuda.get_device_name(0),
                         "precision": "bf16 autocast", "batch": 64, "median_ms": 1e3 * statistics.median(t64),
                         "p90_ms": 1e3 * sorted(t64)[int(0.9 * len(t64))],
                         "images_per_s": 64 / statistics.median(t64)})
            cpu = gpu.cpu().float()
            torch.set_num_threads(max(1, torch.get_num_threads()))
            tc = timed(cpu, x1.cpu(), args.cpu_runs, 5)
            rows.append({"model": kind, "variant": variant, "device": f"CPU ({cpu_name}, {torch.get_num_threads()} "
                         "threads)", "precision": "fp32", "batch": 1, "median_ms": 1e3 * statistics.median(tc),
                         "p90_ms": 1e3 * sorted(tc)[int(0.9 * len(tc))], "images_per_s": 1 / statistics.median(tc)})
            print(f"{kind:13s} {variant:25s} GPU {rows[-3]['median_ms']:6.1f} ms/selfie, "
                  f"{rows[-2]['images_per_s']:7,.0f} img/s in batches; CPU {rows[-1]['median_ms']:7.1f} ms/selfie",
                  flush=True)
            del gpu, cpu, model
            torch.cuda.empty_cache()
    atomic_write(RESULTS / "latency.csv", pd.DataFrame(rows).to_csv(index=False, float_format="%.4g",
                                                                     lineterminator="\n"))
    print("[ OK ] outputs/results/latency.csv")


if __name__ == "__main__":
    main()
