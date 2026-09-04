# Rebuttal experiment protocol

This file is the execution contract for the rebuttal experiments. It separates
confirmatory tests from exploratory analyses and fixes the meaning of every SAE
arm before any new result is interpreted.

## Arm definitions

All controlled arms use the same CLIP checkpoint, hook
(`block_layer=-2`, `module_name=resid`), SAE architecture, target images and
ordering, optimizer, learning-rate schedule, batch size, sparsity coefficient,
training exposure, and evaluation split.

| Arm | Initialization | Activation distribution | Updated parameters |
|---|---|---|---|
| `gsae` | Random, trained on ImageNet | Base CLIP + ImageNet | None during target use |
| `ftsae` | Exact `gsae` checkpoint | LoRA-adapted CLIP + target data | Entire SAE (`protect_frac=0`) |
| `scratchsae` | New random initialization | LoRA-adapted CLIP + target data | Entire SAE |
| `masked` | Exact `gsae` checkpoint | LoRA-adapted CLIP + target data | Unprotected SAE units only |

The old `tasks/train_sae_lora_clip.py` path initializes a new SAE. Therefore,
legacy target-domain checkpoints produced by that script are scratch-trained
checkpoints, even if an older registry calls them `ftsae`. They must not be
used as evidence for checkpoint-initialized fine-tuning without correcting
their provenance.

The trainer's legacy `total_training_tokens` counter increments by input images
(`sae_acts.size(0)`), while each image supplies a full token sequence to the
SAE. New manifests must record both the legacy counter and the derived number
of activation vectors; papers and tables must state which quantity is used.

## 1. Confirmatory MMD predictor

Primary predictor:

- CLIP-space unbiased RBF MMD² between ImageNet and the target domain.
- Fixed feature extractor, bandwidth rule, subsample size, and seed across all
  domains.

Primary outcome:

```text
G-SAE degradation (percentage points)
  = accuracy(LoRA model, no SAE)
  - accuracy(LoRA model, frozen G-SAE reconstruction)
```

This outcome measures the damage caused by reusing the generic dictionary and
does not require choosing the best target SAE after seeing results.

Protocol:

1. Fit the predeclared linear model on the original development domains only.
2. Report in-sample R², leave-one-domain-out Q²/RMSE, and Spearman rank
   correlation. Treat in-sample R² as descriptive, not predictive evidence.
3. Freeze predictions and file hashes before loading outcomes for OfficeHome,
   KITTI, and Cityscapes.
4. Report each held-out prediction with a prediction interval, signed residual,
   held-out MAE/RMSE, rank agreement, and whether the observation lies outside
   the interval.
5. A robust regression is a sensitivity analysis only. It cannot replace the
   primary model after inspecting held-out errors.

OfficeHome is one held-out dataset. Its four domains are concatenated before
the fixed MMD subsample is drawn; they are not four extra test points.

If fewer than the stated 15 development-domain outcomes are present, the tool
may run for debugging, but the result must be labeled exploratory and cannot
support the planned confirmatory claim.

## 2. Causal SAE-feature interventions

Run on LoRA-adapted CLIP for EuroSAT and PathMNIST, comparing `gsae` with the
checkpoint-initialized `ftsae`.

Feature selection and calibration use only the training split. For each
dataset/SAE, select a fixed total of 10--20 class-selective, sufficiently active
features using a rule declared before evaluation. Never select features by
their evaluation-set intervention effect.

For an activation `x`, the intervention is:

1. encode `x` to its natural SAE feature vector `z`;
2. change only the selected coordinate(s);
3. decode the modified `z`;
4. replace the hooked activation with that reconstruction.

The conditions are:

- reconstruction control: decode the unmodified `z`;
- ablation: set selected `z_j` to zero;
- amplification: set selected `z_j` to a training-set-calibrated high value
  (for example, the target-class 95th percentile);
- matched random-feature control: apply the same operation to a feature from
  the same activity-frequency bin.

Do not set every unselected feature to zero or one, and do not subtract a
checkpoint-independent scalar bias.

Primary outcomes are the paired change from reconstruction control in target
class log-probability and target class accuracy. Also report prediction flip
rate, off-target log-probability change, per-feature effects, bootstrap
confidence intervals, and a paired G-SAE versus FT-SAE comparison. A clean
causal result requires FT-SAE effects to exceed its matched controls and the
corresponding G-SAE effects.

## 3. Initialization ablation

Compare `gsae`, checkpoint-initialized `ftsae`, and `scratchsae` on the same
target model and data. The confirmatory comparison is `ftsae` versus
`scratchsae`; `masked` remains a separate stability--plasticity analysis.

Run at least three SAE training seeds on the headline near/mid/far domain
triple. Pair arms by seed and evaluation examples. Record:

- initialization source and source-checkpoint hash;
- model/LoRA checkpoint hash;
- dataset split hash and data-order seed;
- SAE architecture and hook;
- optimizer, schedule, sparsity coefficient, and training exposure;
- final checkpoint hash.

Report mean and a two-sided 95% Student-t confidence interval across independent
SAE training seeds. Evaluation resampling or probe seeds do not count as SAE
training seeds.

## 4. Secondary experiments

These are worthwhile after the three confirmatory experiments above are
complete.

- **EuroSAT specialization:** train/evaluate the same SAE protocol across
  fixed image-count and class-count subsets. Quantify whether the active
  features are class selective and inspect top-activating patches for the
  surviving features.
- **Language-supervision confound:** compare a fixed self-supervised vision
  encoder with a text-aligned counterpart while holding the evaluation and SAE
  protocol fixed. Treat a newly trained lightweight alignment head as a
  different model, not proof about DINOv2 pretraining itself.
- **DAMS robustness:** predeclare EuroSAT and PathMNIST superclass mappings,
  recompute every DAMS arm under the original and coarsened labels, and report
  rank correlation. Add a label-free coherence score as a separate component,
  not a tuned replacement for an unfavorable DAMS result.

## Claim gate

The paper should state a quantitative boundary only after the held-out test:

```text
Below the estimated MMD boundary, frozen G-SAE reuse stays within the declared
degradation tolerance; above it, retraining is recommended.
```

The boundary, tolerance, uncertainty, and exceptions must come directly from
the frozen predictor output. If held-out prediction fails, report the failure
and analyze which domain property is missing rather than choosing a new
boundary post hoc.
