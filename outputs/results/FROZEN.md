# Frozen results


- Analysis rules:
  deviations 1-4 recorded there before any test scoring.
- Runs: clip_lora, clip_probe, effnet_b4 for seeds 0, 1, 2, plus clip_zeroshot (outputs/scores/, not published).
- Frame index content SHA-256: 2a8e44398cc9eb167ac6e0f569a8a251ff475d86054c2f53f97adee71c16ae17.
- Learning rates: outputs/results/lr_choice.json. Analysis: analyze.py, B = 10,000 (2,000 for AUC and re-estimated
  thresholds).
- Independent re-check (23 Oct): H1 type differences, thresholds, Celeb-DF BPCER, H2 DiD and the sanity-check AUC were
  recomputed with separate code (scikit-learn, NumPy) and match exactly.
- Git tag: results-v1.

## Later changes
- exploratory addition (no frozen number changed): fairness_seeds.py repeats the registered fairness
  comparison (Celeb-DF genuine BPCER, female minus male, identity-cluster bootstrap, B = 10,000) for seeds 1 and 2, with
  the same code path as analyze.fairness(); seed 0 reproduces fairness.csv exactly. Output: fairness_seeds.csv. Added
  because the report draft review asked whether the seed-0 gender gaps hold across seeds.
- cosmetic (no number changed): F1_training.png re-rendered so the probe panel's legend no longer covers
  its curves and the validation-AUC axis is ticked only to epoch 20. figures.py regenerated every figure; all others
  are pixel-identical to the frozen versions.
- exploratory addition (no frozen number changed): h2_seeds.py repeats the registered H2
  difference-in-differences (unseen AAR, JPEG 90-30 vs PNG, ViT minus EfficientNet-B4) and its Celeb-DF BPCER and
  unseen-AUC companions for seeds 1 and 2 as point estimates, with the same code path as analyze.h2(); seed 0
  reproduces summary.json exactly. Output: h2_seeds.csv. Added because section 7 calls a result
  robust to training randomness only if the H2 DiD keeps its sign in all three seeds (it does: -6.00, -14.43,
  -5.86 pp). Labelled exploratory in the report.
