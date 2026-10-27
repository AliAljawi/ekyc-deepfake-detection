# Fixed-threshold deepfake detection for bank eKYC: CLIP ViT-B/16 + LoRA vs EfficientNet-B4

Code for the MAAI901 (Deep Learning for Visual Data) Applied Vision Transformer Report, UOWD.

Do fakes the models never saw in training get past an eKYC check? Each detector's threshold is fixed in advance
so that 1% of genuine FF++ validation faces are rejected. The models are then tested on unseen datasets, unseen
generators (DF40) and JPEG compression.

**Main findings**
- H1 rejected: at the fixed threshold, EfficientNet-B4 accepted fewer unseen face swaps and diffusion faces than the
  ViT; reenactment showed no significant difference.
- H2 statistically supported, but as a calibration shift, not robustness: at JPEG quality 60 both models accept
  97-98% of unseen fakes.
- RQ3: LoRA fine-tuning beat a frozen-CLIP linear probe and zero-shot CLIP (which is at chance).
- No significant gender gap in genuine-face rejection, but the intervals are wide.

## Data (not included)
Research-only licences; request each dataset from its authors:
FaceForensics++ c23 (Rössler et al., 2019), Celeb-DF v2 (Li et al., 2020), DF40 (Yan et al., 2024),
DeepfakeBench's split files (Yan et al., 2023) and the demographic annotations of Xu et al. (2024).

Put them in a `datasets` folder next to this repo folder:
```
datasets\
  FaceForensics++\       frames + train.json, val.json, test.json
  Celeb-DF-v2\           frames + List_of_testing_videos.txt
  DF40\                  inswap, simswap, fomm, sadtalker, sd2.1, DiT (each with a cdf\ folder)
  DeepfakeBench_json\    FaceForensics++.json, Celeb-DF-v2.json
  annotations\           A-FF++.csv, A-Celeb-DF.csv
```

## Setup (Windows, Python 3.13, NVIDIA GPU)
```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python check_setup.py
```
`check_setup.py` checks the GPU and downloads the models. All other scripts run offline. On a new PC, run
`$env:HF_HUB_OFFLINE="0"` once before the first `run_all.py` so zero-shot CLIP can download its tokenizer.

## How to run (in order)
| Step | Script | What it does | Time (RTX 5090) |
|---|---|---|---|
| 1 | `build_index.py` | list of all 307,413 frames, checks every count and image | ~3 min |
| 2 | `inspect_data.py` | image statistics and shortcut check | ~2 min |
| 3 | `run_all.py` | learning-rate sweep, training (3 seeds) and scoring | ~3 h |
| 4 | `analyze.py` | every reported number, with cluster-bootstrap 95% CIs | ~5 min |
| 5 | `figures.py` | the report and slide charts | |
| 6 | `explain_maps.py`, then `showcase.py` | attention rollout and Grad-CAM heat maps | |
| 7 | `fairness_seeds.py`, `h2_seeds.py` | exploratory seed 1-2 repeats | |
| optional | `bench_latency.py` | speed per image on GPU and CPU | |

Results go to outputs/results (tables) and outputs/figures (charts) are included; model weights, scores and face images are not.

## Demo
`export_demo.py` exports the seed-0 models, thresholds and 8 test faces, after steps 1-6. Then run
`demo_app.py` (needs `pip install gradio`) or `notebooks/demo_gradio.ipynb` in Google Colab.
Weights and faces are not included.

## AI Use Disclosure
The system architecture, core engineering, and development logic were entirely designed and executed by me. Generative AI (Claude Code by Anthropic) was utilized strictly as a secondary tool for minor tasks, including initial code drafting for specific functions and routine debugging.

## References
- Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-DF: A large-scale challenging dataset for DeepFake forensics. CVPR.
- Rössler, A., et al. (2019). FaceForensics++: Learning to detect manipulated facial images. ICCV.
- Xu, Y., et al. (2024). Analyzing fairness in deepfake detection with massively annotated databases.
- Yan, Z., et al. (2023). DeepfakeBench: A comprehensive benchmark of deepfake detection. NeurIPS.
- Yan, Z., et al. (2024). DF40: Toward next-generation deepfake detection. NeurIPS.
