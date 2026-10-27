# Results B = 10,000


**H1 (joint): rejected.** Celeb-DF real BPCER at the fixed thresholds: ViT 8.1%, CNN 12.5%.

| type | aar_vit | aar_cnn | d_aar | ci_low | ci_high | verdict | p_mcnemar | auc_vit | auc_cnn | oracle_d_aar | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| faceswap | 65.3% | 48.3% | 16.9% | 13.4% | 20.1% | CNN lower | <0.001 | 0.745 | 0.789 | 5.7% | CNN lower |
| reenactment | 88.1% | 89.4% | -1.3% | -3.4% | 0.7% | no significant difference | 1.000 | 0.609 | 0.511 | -0.6% | no significant difference |
| diffusion | 46.5% | 42.6% | 3.9% | 3.7% | 4.2% | CNN lower | <0.001 | 0.779 | 0.773 | 1.6% | CNN lower |

**H2: supported.** DiD -6.00 pp (CI -7.09 to -4.81).

## RQ3

| comparison | type | aar_vit | aar_other | d_aar | ci_low | ci_high | d_auc | d_auc_ci_low | d_auc_ci_high |
|---|---|---|---|---|---|---|---|---|---|
| ViT - Frozen CLIP + probe | faceswap | 65.3% | 92.7% | -27.4% | -31.3% | -23.3% | 0.040 | -0.003 | 0.087 |
| ViT - Frozen CLIP + probe | reenactment | 88.1% | 89.8% | -1.8% | -4.2% | 0.5% | -0.047 | -0.091 | -0.001 |
| ViT - Frozen CLIP + probe | diffusion | 46.5% | 56.7% | -10.2% | -10.5% | -9.8% | -0.031 | -0.054 | -0.007 |
| ViT - Zero-shot CLIP | faceswap | 65.3% | 98.2% | -32.9% | -36.6% | -29.1% | 0.217 | 0.166 | 0.271 |
| ViT - Zero-shot CLIP | reenactment | 88.1% | 99.2% | -11.1% | -13.2% | -9.1% | 0.109 | 0.057 | 0.163 |
| ViT - Zero-shot CLIP | diffusion | 46.5% | 90.4% | -43.9% | -44.2% | -43.6% | 0.058 | 0.026 | 0.092 |

## Fairness (Celeb-DF test real frames, female - male BPCER)

| model | bpcer_female | bpcer_male | gap_f_minus_m | ci_low | ci_high | identities_f | identities_m |
|---|---|---|---|---|---|---|---|
| ViT (CLIP + LoRA) | 9.6% | 7.1% | 2.5% | -1.8% | 6.9% | 53 | 68 |
| EfficientNet-B4 | 10.7% | 13.9% | -3.2% | -9.2% | 2.3% | 53 | 68 |
| Frozen CLIP + probe | 1.3% | 2.2% | -0.9% | -3.3% | 1.6% | 53 | 68 |
| Zero-shot CLIP | 0.7% | 0.0% | 0.7% | 0.0% | 1.6% | 53 | 68 |

## Trade-off

| model | target_val_bpcer | threshold | aar_faceswap | aar_reenactment | aar_diffusion | aar_unseen | bpcer_cdf | bpcer_ffpp_test | review_band_cdf_real | review_band_df40_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT (CLIP + LoRA) | 0.005 | 3.931 | 77.7% | 95.2% | 50.7% | 74.5% | 3.9% | 0.5% | 30.6% | 35.1% |
| ViT (CLIP + LoRA) | 0.010 | 1.780 | 65.3% | 88.1% | 46.5% | 66.6% | 8.1% | 1.1% | - | - |
| ViT (CLIP + LoRA) | 0.020 | -0.073 | 52.6% | 77.7% | 41.5% | 57.3% | 14.5% | 1.9% | - | - |
| ViT (CLIP + LoRA) | 0.050 | -2.829 | 30.9% | 50.9% | 28.8% | 36.9% | 34.5% | 5.6% | - | - |
| EfficientNet-B4 | 0.005 | 5.960 | 64.0% | 95.7% | 47.9% | 69.2% | 5.9% | 0.2% | 36.0% | 33.8% |
| EfficientNet-B4 | 0.010 | 3.568 | 48.3% | 89.4% | 42.6% | 60.1% | 12.5% | 1.0% | - | - |
| EfficientNet-B4 | 0.020 | 1.841 | 36.3% | 80.6% | 37.0% | 51.3% | 20.8% | 1.7% | - | - |
| EfficientNet-B4 | 0.050 | -0.874 | 18.2% | 57.4% | 24.9% | 33.5% | 41.9% | 4.4% | - | - |
| Frozen CLIP + probe | 0.005 | 3.049 | 94.4% | 91.7% | 59.1% | 81.7% | 1.3% | 0.3% | 4.9% | 12.9% |
| Frozen CLIP + probe | 0.010 | 2.651 | 92.7% | 89.8% | 56.7% | 79.7% | 1.8% | 0.4% | - | - |
| Frozen CLIP + probe | 0.020 | 1.974 | 88.8% | 86.0% | 52.6% | 75.8% | 3.0% | 1.2% | - | - |
| Frozen CLIP + probe | 0.050 | 0.994 | 81.0% | 78.3% | 46.8% | 68.7% | 6.2% | 4.2% | - | - |
| Zero-shot CLIP | 0.005 | 3.897 | 98.5% | 99.4% | 92.2% | 96.7% | 0.2% | 0.9% | 1.1% | 4.8% |
| Zero-shot CLIP | 0.010 | 3.741 | 98.2% | 99.2% | 90.4% | 95.9% | 0.3% | 1.4% | - | - |
| Zero-shot CLIP | 0.020 | 3.610 | 97.9% | 98.9% | 88.7% | 95.2% | 0.4% | 1.8% | - | - |
| Zero-shot CLIP | 0.050 | 3.290 | 96.2% | 98.0% | 83.4% | 92.5% | 1.3% | 4.1% | - | - |

## Seeds

| seed | model | aar_faceswap | aar_reenactment | aar_diffusion | aar_unseen | bpcer_cdf | aar_cdf | auc_cdf | threshold | val_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ViT (CLIP + LoRA) | 65.3% | 88.1% | 46.5% | 66.6% | 8.1% | 64.4% | 0.771 | 1.780 | 0.990 |
| 0 | EfficientNet-B4 | 48.3% | 89.4% | 42.6% | 60.1% | 12.5% | 67.1% | 0.702 | 3.568 | 0.990 |
| 0 | Frozen CLIP + probe | 92.7% | 89.8% | 56.7% | 79.7% | 1.8% | 89.8% | 0.731 | 2.651 | 0.862 |
| 0 | ViT - CNN | 16.9% | -1.3% | 3.9% | 6.5% | - | - | - | - | - |
| 1 | ViT (CLIP + LoRA) | 77.3% | 94.7% | 50.8% | 74.3% | 4.4% | 69.9% | 0.792 | 3.027 | 0.989 |
| 1 | EfficientNet-B4 | 37.9% | 77.7% | 35.0% | 50.2% | 20.8% | 59.7% | 0.665 | 3.069 | 0.990 |
| 1 | Frozen CLIP + probe | 94.4% | 91.8% | 55.8% | 80.7% | 1.6% | 93.0% | 0.695 | 2.952 | 0.858 |
| 1 | ViT - CNN | 39.4% | 17.0% | 15.8% | 24.1% | - | - | - | - | - |
| 2 | ViT (CLIP + LoRA) | 67.2% | 87.5% | 48.0% | 67.6% | 8.1% | 50.3% | 0.827 | 1.961 | 0.991 |
| 2 | EfficientNet-B4 | 50.6% | 95.8% | 44.0% | 63.5% | 8.3% | 77.7% | 0.690 | 2.480 | 0.990 |
| 2 | Frozen CLIP + probe | 92.8% | 90.3% | 54.5% | 79.2% | 2.0% | 91.6% | 0.694 | 2.652 | 0.857 |
| 2 | ViT - CNN | 16.6% | -8.3% | 4.0% | 4.1% | - | - | - | - | - |

## Sanity check

| seed | effnet_cdf_frame_auc | band | reference |
|---|---|---|---|
| 0 | 0.702 | consistent with 0.7487 | 0.749 |
| 1 | 0.665 | consistent with 0.7487 | 0.749 |
| 2 | 0.690 | consistent with 0.7487 | 0.749 |
