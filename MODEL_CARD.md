# Model card: frame-level deepfake detectors for eKYC (research prototypes)

Format after Mitchell et al. (2019), "Model Cards for Model Reporting". All numbers are seed 0, original PNG, at each
model's fixed threshold, from `outputs/results/` (analysis rules: `ANALYSIS_PLAN.md`).

## Models
| | ViT (main) | Baseline | Probe | Zero-shot |
|---|---|---|---|---|
| Backbone | CLIP ViT-B/16 (`openai/clip-vit-base-patch16`) | EfficientNet-B4 (`timm efficientnet_b4.ra2_in1k`) | CLIP ViT-B/16, frozen | CLIP ViT-B/16 + text encoder |
| Adaptation | LoRA r=16 on q/k/v/out projections + linear head (1.18 M trainable) | full fine-tuning (17.6 M) | linear head (769) | none (text prompts) |
| Input | 224x224 face crop | 256x256 face crop | 224x224 | 224x224 |
| Output | one logit; "fake" if above the fixed threshold | same | same | same |

## Intended use
One signal among several in remote onboarding (eKYC): scoring a single face crop for signs of manipulation, next to
liveness detection, document checks and face matching, with human review of borderline or rejected cases. Research
and teaching only (MAAI901, UOWD, 2026).

## Out of scope
- Sole or automatic decision-maker for opening or refusing an account.
- Surveillance, identifying people, or judging any attribute of a person.
- Images that are not tightly cropped, aligned faces like the DeepfakeBench crops (webcam frames, documents, full photos).
- Commercial use: the training data (FaceForensics++) is licensed for research only.

## Training data
FaceForensics++ c23 (Rössler et al., 2019), DeepfakeBench face crops (Yan et al., 2023), official train split: real
YouTube videos and four manipulations (Deepfakes, Face2Face, FaceSwap, NeuralTextures). Model selection and the
threshold use the official validation split only.

## Evaluation data
FF++ test (seen methods), Celeb-DF v2 test (Li et al., 2020), and six unseen DF40 methods in the Celeb-DF domain
(Yan et al., 2024): InSwapper, SimSwap (face swap), FOMM, SadTalker (reenactment), SD-2.1, DiT-XL/2 (diffusion), each
at the original quality and JPEG 90/75/60/45/30.

## Results (attack acceptance rate AAR = fakes accepted; BPCER = genuine rejected)
| | ViT | EfficientNet-B4 | Probe | Zero-shot |
|---|---|---|---|---|
| FF++ test BPCER (threshold target 1%) | 1.1% | 1.0% | 0.4% | 1.4% |
| Celeb-DF test BPCER | 8.1% | 12.5% | 1.8% | 0.3% |
| Celeb-DF test AUC | 0.771 | 0.702 | 0.731 | 0.495 |
| AAR face swap / reenactment / diffusion | 65 / 88 / 47% | 48 / 89 / 43% | 93 / 90 / 57% | 98 / 99 / 90% |
| Unseen AAR at JPEG quality 60 | 98% | 97% | - | - |
| Latency per selfie, GPU (RTX 5090) / desktop CPU | 5 ms (LoRA merged) / 66 ms | 12 ms / 19 ms | 5 / 66 ms | 5 / 67 ms |

## Fairness
Genuine Celeb-DF faces rejected, female minus male (apparent gender, classifier-predicted labels of Xu et al., 2024;
identity-cluster 95% CI): ViT +2.5 pp [-1.8, 6.9], EfficientNet -3.2 pp [-9.2, 2.3]. No significant gap, but the test
set (53 female and 68 male identities) cannot rule out gaps of up to 7 pp (ViT) or 9 pp (EfficientNet). Exploratory,
seeds 1-2 (`outputs/results/fairness_seeds.csv`): ViT +2.1 and +0.7 pp, EfficientNet +3.8 and -1.4 pp, none significant.
Ethnicity could not be assessed: of 178 genuine Celeb-DF test videos, 142 are labelled White, 3 Black and 33
undefined; FF++ test has 1 Black genuine video; neither label set has an Arab or South Asian category.

## Caveats and limitations
- The threshold does not transfer: set for 1% genuine rejection on FF++, it rejects 8-12.5% of genuine Celeb-DF faces.
  A deployment must recalibrate on its own genuine customers and monitor drift.
- Unseen face swaps, reenactments and diffusion faces are mostly accepted; JPEG compression (quality 60 or lower)
  defeats both models almost completely. No compression augmentation was used in training.
- The faces are celebrities and YouTube videos, not onboarding selfies; CLIP may have seen some of these people in
  pretraining.
- SD-2.1 and DiT images differ from real frames in simple image statistics, so their results may partly reflect image
  quality rather than manipulation.

## Before any real use
Consented, representative training and calibration data (including UAE customers, with self-reported demographics);
a demographic-differential audit (ISO/IEC 19795-10); compression-robust training; recalibration and drift monitoring;
human review and an appeal route for rejected customers; layering with liveness, injection and document checks.
