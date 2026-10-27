# Analysis plan

Before any model was trained: the learning-rate sweep starts.

Already known: dataset composition and the checks in `outputs/results/` (index summary, image
statistics, one-feature shortcut AUCs, loader speed). No model outputs on any test set existed.

## 1. Question, metrics and hypotheses

Setting: frame-level, reference-free deepfake detection for remote onboarding (eKYC), where the decision threshold
must be fixed before new attacks appear.

- **AAR** (attack acceptance rate): share of fake frames classified as genuine at the fixed threshold
  (= 1 - recall on fakes).
- **BPCER** (bona fide presentation classification error rate): share of real frames classified as fake at the fixed
  threshold (= 1 - specificity). AAR/BPCER follow ISO/IEC 30107-3 (APCER/BPCER); here the genuine error is fixed and
  the attack error measured.
- **Threshold:** fixed per model on FF++ validation real frames so that BPCER = 1% (section 4).

Approaches compared (all trained identically, section 3):

| Name | Model | Adaptation | Input |
|---|---|---|---|
| ViT (CLIP-ViT approach) | CLIP ViT-B/16 image encoder, `openai/clip-vit-base-patch16` | LoRA r=16, alpha=16, dropout 0.1 on q/k/v/out projections; linear head on the pooled CLS token | 224x224, CLIP mean/std |
| CNN (baseline) | EfficientNet-B4, `timm efficientnet_b4.ra2_in1k` (ImageNet-1k) | full fine-tuning | 256x256, ImageNet mean/std |
| Probe | the same CLIP encoder, frozen | linear head on the pooled CLS token (embedding extraction) | 224x224 |
| Zero-shot CLIP | the same CLIP model with its text encoder | none (text prompts, section 4) | 224x224 |

- **H1.** For each unseen attack type - face swap (InSwapper, SimSwap), reenactment (FOMM, SadTalker) and diffusion
  (SD-2.1, DiT-XL/2), all from DF40's Celeb-DF domain - the ViT has a lower AAR than the CNN.
  **H0 (per type):** no difference in AAR.
- **H2.** Averaged over JPEG quality 90, 75, 60, 45 and 30, the ViT's AAR increases less (relative to the original
  images) than the CNN's. **H0:** it does not.
- **RQ3 (secondary, descriptive).** Fine-tuning (ViT) vs embedding extraction (probe) vs zero-shot CLIP on the same
  unseen attacks.
- **Fairness question (secondary).** Are genuine faces rejected at different rates by apparent gender?

The two approaches differ in more than architecture (pretraining data and objective, LoRA vs full fine-tuning,
input size), so results compare approaches, not architectures alone.

## 2. Data

| Role | Data | Videos (frames) |
|---|---|---|
| Training | FaceForensics++ c23, DeepfakeBench crops, official train split: real, Deepfakes, Face2Face, FaceSwap, NeuralTextures | 719 / 719 / 719 / 720 / 719 (114,884) |
| Selection and threshold | FF++ official val split | 140 per class (22,354; 4,477 real) |
| Test, seen methods | FF++ official test split | 140 per class (22,388) |
| Test, unseen | Celeb-DF v2 official test list | 178 real + 340 fake (16,420) |
| Test, unseen (H1) | DF40, Celeb-DF domain: InSwapper, SimSwap, FOMM, SadTalker, SD-2.1, DiT-XL/2 | 424 / 641 / 647 / 660 / 888 / 888 folders (131,367) |

- Celeb-DF and DF40 are never used for training, model selection or threshold setting.
- The frame index is `outputs/index/frames.csv.gz` (not redistributed). Every run records it.
- SD-2.1 images (512x512) are resized to 256x256 (bicubic, antialiased) before anything else.
- Apparent gender: Xu et al. (2024) classifier-predicted labels, joined per video (constant within each video). DF40
  swap/reenactment folders inherit the gender of their Celeb-DF identities only where target and source agree
  (SadTalker: the target); diffusion images have no label.

## 3. Training protocol (identical for ViT, CNN and probe)

- **Frames per epoch:** 16 random frames per real video and 4 per fake video (all frames if fewer): 23,012 frames
  (real 11,504; Deepfakes 2,876; Face2Face 2,876; FaceSwap 2,880; NeuralTextures 2,876), shuffled. The frames, their
  order and every augmentation draw depend only on (seed, epoch) and are identical across models; a per-epoch hash is
  logged and compared.
- **Augmentation:** horizontal flip (p = 0.5); RandomResizedCrop to 256 (scale 0.8-1.0, ratio 0.95-1.05);
  brightness and contrast jitter of 0.1 (p = 0.5). No blur and no JPEG, so compression stays unseen.
- **Optimisation:** AdamW (betas 0.9/0.999), weight decay 0.01 on weight matrices only, batch 64, 20 epochs of 359
  steps, linear warm-up over the first 5% of steps then cosine decay to 0, bf16 autocast with the final linear layer in
  fp32, gradient-norm clipping at 1.0. New linear heads use learning rate 1e-3, except the probe, whose head uses the
  swept rate.
- **Learning-rate sweep** (seed 0, 4 epochs, the same schedule shape): ViT {1e-4, 3e-4, 1e-3}; CNN {5e-5, 2e-4,
  5e-4}; probe {1e-3, 3e-3, 1e-2}. If the selected rate is at an edge of its grid, one extra rate 3x beyond that edge
  is run once, and the rule below is applied again over all four.
- **Selection rule** (for epochs within a run, and for learning rates using each rate's selected epoch): candidates
  are those whose FF++ val AUC is less than 0.001 below the best; among them, the lowest balanced val loss
  (0.5 x mean BCE on real frames + 0.5 x the mean over the four methods of each method's mean BCE); remaining ties go
  to the earlier epoch or the lower learning rate. Only FF++ val is used.
- **Seeds** 0, 1 and 2. Seed 0 is the primary run for every hypothesis test; seeds 1-2 show robustness to training
  randomness.
- **Gate before the main runs:** stop if any loss is non-finite or a model's best sweep val AUC is below 0.60.
- **No hyper-parameter changes after any test set has been scored.**

## 4. Scoring and threshold

- Each trained model (and zero-shot CLIP) scores FF++ val at the original PNG and every test frame at six levels:
  original PNG and JPEG quality 90, 75, 60, 45, 30 (libjpeg-turbo, 4:2:0 chroma subsampling, standard tables),
  applied to the 256x256 crop, to real and fake frames alike.
- **Threshold** per model and seed: the smallest FF++ val real logit t such that at most 1% of the 4,477 val real
  frames lie strictly above it (t is the 4,433rd smallest; 44 frames, 0.98%, lie above). A frame is called fake iff
  its logit > t. The threshold is never re-estimated on test data, except in the labelled "oracle" analysis (5.3).
- **Zero-shot prompts** (fixed here, before scoring). Real: "a photo of a real human face", "a real photograph of a
  person". Fake: "a deepfake image of a face", "a fake, AI-generated face". Score = CLIP's logit scale x (mean cosine
  similarity of the projected image embedding to the fake prompts - mean cosine similarity to the real prompts);
  threshold by the same 1% rule.

## 5. H1 (primary)

### 5.1 Scope and statistic
- DF40 methods, seed 0, original PNG. A type's AAR is the mean of its two methods' AARs. Celeb-DF test is reported
  next to H1, not inside it. Claims are worded "for these two methods of each type".
- Per type: dAAR = AAR(ViT) - AAR(CNN), with a paired cluster-bootstrap percentile 95% CI (B = 10,000; section 9).

### 5.2 Verdicts
- Per type: "ViT lower" if the CI's upper bound < 0; "CNN lower" if its lower bound > 0; otherwise "no significant
  difference". If the two methods of a type have dAARs of opposite sign and both method-level CIs exclude 0, the type
  is "method-dependent".
- **Joint H1** is supported only if all three types are "ViT lower" (intersection-union test, Berger 1982: each
  two-sided 95% CI is a one-sided 2.5% test). Otherwise H1 is rejected. When the joint H1 fails, per-type claims use
  Holm-adjusted p-values (Holm 1979), applied separately to the bootstrap and the McNemar p-values.
- **McNemar's test** (exact two-sided binomial test on discordant pairs; Dietterich 1998) per type, on paired ViT vs
  CNN decisions with one unit per source clip: for face swap and reenactment, the Celeb-DF target clip (`idX_NNNN`),
  accepted if more than 50% of that clip's frames from both methods of the type are accepted (a tie counts as
  rejected); for diffusion, one unit per image. It is reported next to the bootstrap CI. **The bootstrap CI decides**;
  if the two disagree, the report says so and why (majority-vote units, discordant units only, low power when both
  AARs are high).

### 5.3 Reading rule (calibration guard)
dBPCER_CDF = BPCER(ViT) - BPCER(CNN) on Celeb-DF test real frames at the fixed thresholds (paired CI). Oracle dAAR =
dAAR with each model's threshold re-set to 1% BPCER on Celeb-DF test real frames, recomputed inside every bootstrap
replicate (B = 2,000), labelled "oracle, not available to a bank".

| dAAR CI | dBPCER_CDF CI | Oracle dAAR CI | Reported as |
|---|---|---|---|
| < 0 | contains 0, or < 0 | any | the ViT lets fewer attacks through at no extra cost to genuine customers |
| < 0 | > 0 | < 0 | the ViT separates better but also rejects more genuine faces (a trade-off) |
| < 0 | > 0 | contains 0, or > 0 | the gain is a calibration shift, not better detection |
| contains 0 | - | - | no significant difference |
| > 0 | - | - | the CNN lets fewer attacks through |

H1 counts as supported only if every type falls in rows 1-3; rows 2-3 are reported as qualified support. Each type's
AUC (mean of its two methods' AUCs, fakes against Celeb-DF test real frames) is shown next to its AAR.

## 6. H2

- U(m, q) = the overall unseen AAR of model m at level q: the mean of the three type AARs (DF40, seed 0).
- **Statistic:** DiD = mean over q in {90, 75, 60, 45, 30} of [(U(ViT, q) - U(ViT, png)) - (U(CNN, q) - U(CNN, png))]
  (the area between the two degradation curves), with a paired bootstrap 95% CI from the same replicates.
- **Verdict:** H2 is supported if the CI's upper bound < 0; otherwise rejected.
- **Companions** (same replicates): dAAR at q = 30; the same DiD for BPCER on Celeb-DF test real frames; the same DiD
  for AUC. "Degrades less" is read as robustness only if the BPCER DiD's CI is not entirely above 0 and the AUC DiD
  points the same way. If H2 fails but dAAR at q = 30 is < 0 (CI), the report says "degrades more but stays better".
- Retention (the share of fakes caught at PNG that are still caught at q = 30) is reported in the repository.

## 7. Secondary analyses

- **RQ3:** paired ViT - probe and ViT - zero-shot dAAR and dAUC with CIs, per type, seed 0, PNG. Descriptive: no
  formal hypothesis and no verdict beyond the CIs.
- **Fairness:** BPCER on Celeb-DF test real frames, female minus male (percentage
  points), original PNG, seed 0, per model, at the single global threshold, with an identity-cluster bootstrap CI;
  plus the ratio and the number of identities and videos per group. Everything else is exploratory: AAR by gender
  (Celeb-DF fakes; DF40 inherited labels), FF++ test by gender and age. Ethnicity is not analysed (1-3 Black faces per
  test set; no Asian label in Celeb-DF; no Arab or South Asian label anywhere); the report states this.
- **Threshold trade-off:** AAR and Celeb-DF BPCER at thresholds set to FF++-val BPCER of 0.5%, 1%, 2% and 5%, and the
  share of Celeb-DF real frames and DF40 fake frames between the 0.5% and 5% thresholds (a "human review band").
- **Threshold uncertainty:** inference is conditional on the fixed threshold (the bank's situation). The report states
  how many of the 140 val real videos contribute the 44 frames above the threshold, gives the FF++ test real BPCER
  with its CI, and says whether any H1 verdict changes when the threshold is re-estimated inside every replicate
  (resampling the 140 val real videos). Verdicts are called threshold-robust only if none changes.
- **Seeds:** all three per-seed values are shown (each seed with its own threshold); no mean +/- SD from n = 3. A
  result is robust to training randomness if dAAR (and the H2 DiD) has the same sign in all three seeds for every
  type; otherwise the type that flips is named. The CIs and McNemar's test cover sampling of test clips, not training
  randomness.
- **Standard metrics** at the same threshold, for FF++ test and Celeb-DF test (both classes present): confusion
  matrix, accuracy, precision, recall, specificity, F1 (fake = positive). DF40 is fake-only: AAR per method, and AUC
  against Celeb-DF test real frames.

## 8. Sanity check against DeepfakeBench

DeepfakeBench reports a frame AUC of 0.7487 for EfficientNet-B4 trained on FF++ c23 and tested on Celeb-DF v2,
selected as the best of about 21 test-set evaluations, with JPEG/blur augmentation and different weights and
normalisation. Bands for the CNN's Celeb-DF frame AUC (PNG, seed 0):

| AUC | Reading |
|---|---|
| 0.65-0.80 | consistent with 0.7487 |
| 0.60-0.65 | lower than expected |
| 0.80-0.85 | higher than expected; a leakage check is run |
| < 0.60 or > 0.85 (any seed) | only checks that do not use test scores (label map, normalisation, resize, checkpoint round trip, val AUC); a confirmed bug is fixed for every model, all runs are redone, and both results are reported |

The report says "consistent with", never "near" or "replicates".

## 9. Bootstrap

- **Units (clusters).** FF++ test: the official pair (70; each with 2 real and 8 fake videos). FF++ val real: the
  video (140). Celeb-DF test real: the video, named by its clip (`idX_NNNN`, or the YouTube-real video). Celeb-DF test
  fake and DF40 face swap/reenactment: the target clip `idX_NNNN` (one shared namespace with the Celeb-DF real clips).
  DF40 diffusion: the image (the folders are random batches of unrelated images, so these CIs are probably
  optimistic). Gender analyses: the identity. Sensitivity: identity clusters for everything.
- **Strata,** each resampled independently and shared by every quantity: CDF (Celeb-DF test clips and the four DF40
  video methods), FF (70 test pairs), SD (28,174 images), DIT (28,174 images), VAL (140 val real videos; threshold
  uncertainty only), plus identity strata for gender and sensitivity.
- **Replicates:** for stratum s with C clusters, chunk c draws `idx = default_rng([SEED, s, c]).integers(0, C, (b, C))`
  and converts it to multinomial count weights W (b x C). Rates are ratio estimators: AAR* = (W @ accepted counts) /
  (W @ frame counts) per cluster. AUC uses the frames' cluster weights in a weighted Mann-Whitney statistic (ties
  1/2). Weighted thresholds (oracle, sensitivity) use the same 99% rule on weighted counts. B = 10,000 for rates and
  differences; B = 2,000 for AUC and anything that re-estimates a threshold. Unit tests: W = 1 reproduces every point
  estimate; the weighted AUC equals scikit-learn's.
- **CIs:** percentile 95%. **p-values:** two-sided bootstrap p = min(1, 2 x min(1 + #{D* >= 0}, 1 + #{D* <= 0}) /
  (B + 1)); H2 one-sided p = (1 + #{DiD* >= 0}) / (B + 1); McNemar: exact binomial on b discordant units out of b + c
  (p = 1 if b + c = 0).

## 10. Reporting

- Effect sizes in percentage points and as attacks accepted per 1,000 attempts. With about 28,000 images per
  diffusion method, tiny gaps can be "significant"; the size is always reported with the CI.
- The analysis code (`analyze.py`) computes every reported number from the stored scores; results are frozen with a
  git tag before the report is written.

## 11. Deviations

during the learning-rate sweep, before any model had scored a test frame.
They came from the independent code review run in parallel with the sweep.

1. **EfficientNet-B4 head initialisation (training).** timm initialised the new 1-output classifier with
   U(-1, 1) (TensorFlow-style 1/sqrt(fan_out) with fan_out = 1): about 27 times the scale of the ViT's and the
   probe's heads, which use PyTorch's default Linear initialisation (U(+/-1/sqrt(fan_in))). The untrained CNN
   started from logits with a standard deviation of about 2.5 instead of about 0.04, so the three models did not start
   from equivalent heads. Changed to PyTorch's default initialisation, as for the other heads and as in
   DeepfakeBench's EfficientNet-B4. EfficientNet's sweep runs made with the old initialisation were discarded
   and re-run; the ViT and probe sweep runs were unaffected and kept.
2. **FOMM bootstrap unit (analysis).** Section 9 assumed DF40 folders `idX_idY_NNNN` show the target clip
   `idX_NNNN`. That holds for InSwapper, SimSwap and SadTalker, but FOMM animates a still of clip `idY_NNNN` with
   the motion of `idX_NNNN`: FOMM frames are closer in pixels to `idY_NNNN` than to `idX_NNNN` in 11 of 11 checked
   folders (and in 35 of 40 in the review's own check). Following section 9's principle (the unit is the clip a
   fake is made from), FOMM's cluster and identity are now `cdf/idY_NNNN` and `idY`. Only these two columns
   changed when the frame index was rebuilt (no training frame changed); . The gender rule is unchanged (FOMM
   keeps a label only where both identities agree).
3. **Gate enforcement (orchestration).** A non-finite loss now stops the chain immediately; previously the job
   would have been retried once, which could have hidden a gate failure. (The sweep log so far had no non-finite
   loss.)
4. **Scope of the non-finite-loss gate (protocol clarification, decided at 7:15 PM, while EfficientNet's re-run sweep
   was at its first rate and before any of its results).** A learning-rate *candidate* whose loss becomes non-finite
   during the sweep is recorded as "diverged" and cannot be selected, in the same way the collapsed ViT run at 3e-3
   (val AUC 0.51) lost on the selection rule. The gate still stops everything if a *selected* rate's run, or any main
   run (20 epochs, any seed), has a non-finite loss. If the extra edge rate diverges, the edge rate stays selected,
   and the report says it sits at the edge of the explored range.
