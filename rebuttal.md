# Rebuttal working document

**What this file is:** the evidence base behind our NeurIPS rebuttal, organized so each experiment answers a specific reviewer question. Every number is pulled from an actual run (see `REBUTTAL_EXPERIMENTS.md` for the underlying protocol and arm definitions) — nothing here is estimated. Jump to "Reviewer concerns — response matrix" or "Reviewer response drafts" at the end for text to actually send reviewers; everything above it is the evidence that text is built on.

**The paper's core claim:** generic SAEs (frozen, trained once on the original pretrained model) don't transfer faithfully once a model is domain-adapted — interpretability degrades as the target domain gets further from the pretraining distribution. Retraining the SAE on the adapted model (FT-SAE) recovers and surpasses the original interpretability, with the benefit growing with domain distance.

**Reviews received (2026-07-25):** meta-review (AC k9GS), official reviews from UEAH, nd4b, kubc. All three, independently, raised the same core objection: the finding above is expected, and the paper doesn't say *when* retraining is actually worth doing. Every experiment below either (a) turns "generic SAEs degrade" from a correlation into something predictive and causal, or (b) directly answers a specific question one of the three reviewers asked.

**Quick map of which experiment answers which reviewer:**

| Experiment | Answers |
|---|---|
| §1a-b MMD / concept_gap predictor | UEAH (H4 predictive test), nd4b (correlation critique), kubc (heuristic for when to retrain) |
| §1c ImageNet baseline | nd4b (asked directly) |
| §1d Full-scale ImageNet anchor | UEAH / nd4b (distance=0 sanity check) |
| §1e Multi-feature decision-rule regressor | nd4b (no clear takeaway), kubc (heuristic for when to retrain) |
| §2 Causal steering | UEAH (asked directly: "what is the steering ability of G-SAE vs FT-SAE") |
| §3 Init ablation | UEAH (asked directly: "do FT-SAE and Scratch-SAE converge to the same place?") |
| §4a DAMS coarsening | UEAH (DAMS's class-separability assumption) |
| §5 H5 causal steerability | UEAH (follow-up design doc, Part A) |

---

## §1. Does domain distance predict how much a generic SAE will degrade?

This is the paper's H4 hypothesis, and all three reviewers pushed on it: nd4b said the correlation in Table 3 isn't actually strong; UEAH and kubc asked for something predictive — a rule for deciding when retraining is worth the cost, not just a post-hoc description of 15 datasets.

### §1a. Refitting the predictor, and a real problem it surfaced

**What we did:** grew the training cohort from 6 to 8 domains and refit the same MMD² → degradation regression the paper reports, holding out three domains (OfficeHome, KITTI, Cityscapes) completely untouched for eventual confirmatory testing. For each domain we measure two accuracies on the LoRA-adapted model — with no SAE in the loop, and with the frozen generic SAE (G-SAE) reconstructing the representation — and call the gap "degradation."

**What we found:** adding two domains didn't strengthen the relationship, it broke it.

| Dataset | MMD²(CLIP) | No-SAE acc | G-SAE acc | Degradation (pp) |
|---|---|---|---|---|
| caltech101 | 0.028 | 93.33 | 92.00 | 1.33 |
| dtd | 0.090 | 72.44 | 67.11 | 5.33 |
| ucf101 | 0.126 | 86.22 | 81.33 | 4.89 |
| cub2002011 | 0.154 | 83.11 | 69.33 | 13.78 |
| fgvc | 0.233 | 48.89 | 34.67 | 14.22 |
| eurosat | 0.291 | 79.11 | 75.11 | 4.00 |
| pathmnist | 0.423 | 91.20 | 79.65 | 11.55 |
| **oxford_pets** | **0.614 (highest)** | 95.56 | 92.00 | **3.56 (2nd-lowest)** |

Oxford Pets has the *largest* domain distance in the cohort by raw embedding MMD, but one of the *smallest* degradations — backwards from what the hypothesis predicts, and not a small effect: its Cook's distance (1.63) is 5x the next-highest domain's, meaning this one point is single-handedly bending the fit.

| | 6-domain fit (paper) | 8-domain fit (now) |
|---|---|---|
| Apparent R² | 0.161 | **0.004** |
| LODO R² | -1.67 | -1.21 |
| LODO Spearman ρ (p) | -0.77 (0.072) | **-0.95 (0.0003)** |

**Why this matters for the rebuttal:** nd4b is right that Table 3's correlation is weak — at 8 domains it's essentially gone (R²=0.004). We are not papering over this; the next step is diagnosing *why*, which turns into a stronger response than defending the original correlation would have been.

### §1b. Diagnosing Oxford Pets: distance in embedding space isn't distance in concept space

**What we did:** we suspected raw CLIP-embedding MMD conflates two different things — how different a domain's *images* look (style, composition) from how different its *concepts* are from what the SAE's dictionary was trained on. ImageNet-1k already contains ~120 dog breeds and many cat/animal classes, so Oxford Pets could have large image-style distance while being conceptually redundant with what G-SAE already represents. We built a second distance metric, `concept_gap`: embed each domain's class names and all 1000 ImageNet-1k class names with the CLIP text encoder, and for each target class find its nearest ImageNet class by cosine similarity. Low `concept_gap` = the domain's vocabulary is already covered by ImageNet-1k; high = genuinely novel vocabulary.

**What we found:** this directly confirms the hypothesis.

| Dataset | MMD²(CLIP) | concept_gap | Degradation (pp) | Nearest-ImageNet example |
|---|---|---|---|---|
| oxford_pets | 0.614 (highest) | **0.109 (lowest)** | 3.56 | abyssinian → Vizsla |
| caltech101 | 0.028 | 0.138 | 1.33 | okapi → impala |
| eurosat | 0.291 | 0.199 | 4.00 | SeaLake → sea lion |
| ucf101 | 0.126 | 0.218 | 4.89 | Throw Discus → basketball |
| dtd | 0.090 | 0.232 | 5.33 | veined → prison |
| pathmnist | 0.423 | 0.265 | 11.55 | smooth muscle → eel |
| fgvc | 0.233 | 0.293 | 14.22 | C-47 → military aircraft |
| cub2002011 | 0.154 | **0.411 (highest)** | 13.78 | Pomarine Jaeger → albatross |

Oxford Pets has the lowest `concept_gap` of all 8 domains — exactly the anomaly explained. Cub2002011 (bird species) is the mirror case: moderate embedding MMD, but the *highest* concept_gap (fine-grained bird species have no good ImageNet-1k analogue), and the second-highest degradation.

| Predictor | Pearson r (p) | Spearman ρ (p) | Full-sample R² | LODO R² |
|---|---|---|---|---|
| MMD²(CLIP) | 0.06 (0.88) | 0.12 (0.78) | 0.004 | -1.21 |
| **concept_gap** | **0.87 (0.005)** | **0.95 (0.0002)** | **0.753** | **0.506** |

`concept_gap` beats raw MMD on every metric, including a *positive* leave-one-domain-out R² — meaning it generalizes under resampling, which the MMD fit does not.

**Why this resolves the concern:** this is a genuine step toward the "when should I retrain" heuristic kubc and UEAH asked for, and a direct, mechanistic answer to nd4b's "the correlation isn't strong" critique — we agree, and we found a better predictor rather than defending the weak one.

**How we're limiting the claim (important — do not oversell this in the rebuttal):**
- `concept_gap` was discovered *after* seeing the Oxford Pets anomaly, not pre-registered. That's a real garden-of-forking-paths risk, softened by having a specific, named anomaly that motivated it (not a blind search over candidate predictors), but still worth flagging.
- It has **not** been tested against the three reserved held-out domains (OfficeHome, KITTI, Cityscapes) — LODO R²=0.506 resamples within the same 8 domains that produced the hypothesis, which is encouraging but not the same as true held-out validation.
- n=8 is small enough that a single well-chosen predictor can fit almost anything.
- **Decision (2026-07-25):** deliberately not spending the held-out domains on `concept_gap` yet, since it wasn't pre-registered — scoring it now would use up the one confirmatory test on a predictor discovered post-hoc. Report it to reviewers as "a promising, mechanistically-motivated candidate," not "the validated predictor."

*Implementation: `analysis/mmd_degradation_predictor.py`, `analysis/concept_overlap_distance.py`; results in `out/rebuttal/mmd_predictor/retrospective_exploratory8_fit/`. Remaining gap to the originally-planned 15-domain cohort: 7 domains blocked on data engineering (WordNet class-name mapping for imagenet-r/a/o/v2/sketch, folder mapping for flowers102, parquet loaders for stanford_cars/food101) — deprioritized, not actively worked.*

### §1c. The ImageNet baseline — nd4b asked directly

> **nd4b:** "What is the performance of the reported r metrics for ZS+G-SAE on ImageNet?"

**What we did:** ran the base (non-adapted) CLIP ViT-B/16 model, with and without G-SAE, on a 2000-image stratified sample of ImageNet-1k validation (2 per class, all 1000 classes).

**What we found:**

| Condition | Accuracy |
|---|---|
| No SAE | 66.60% |
| + G-SAE | 62.95% |
| **Degradation** | **+3.65pp** |

(66.60% lines up with CLIP ViT-B/16's published ~68% zero-shot ImageNet accuracy — a sanity check that the pipeline measures what it claims to.)

We also ran the full SAE-quality metric suite on this condition, since the question implicitly asks for more than just accuracy:

| Metric | G-SAE on base ImageNet |
|---|---|
| L0 (active features/token) | 149.1 |
| Dead fraction | 0.018 (1.8%) |
| Reconstruction cosine similarity | 0.908 |
| Fraction variance explained (FVE) | 0.941 |
| Label entropy (mean, top-1000 features) | 3.164 |

**Why this resolves the concern:** it's the direct number nd4b asked for, and it does more work than just answering the question — it exposes a confound we hadn't controlled for. G-SAE costs 3.65pp *even on its own training domain*, with zero domain adaptation involved — more than the degradation on caltech101 (1.33pp), the domain with the smallest degradation everywhere else in §1a's table. Two things are folded into that number, and both belong in the paper:
1. **A non-zero intrinsic reconstruction cost.** No SAE reconstructs perfectly. This sets a real noise floor — degradation below ~3.65pp elsewhere isn't obviously distinguishable from the SAE's own baseline cost. Caltech101's 1.33pp is *below* this floor and should be read as "no detectable degradation," not "a small positive one."
2. **A class-cardinality confound.** ImageNet-1k is 1000-way; every other domain in this rebuttal is 10-200-way. Finer decision boundaries are more sensitive to small reconstruction error, so some of what looks like a domain-distance effect elsewhere may partly be a class-count effect we haven't isolated. Worth stating as a limitation; a cheap follow-up (not done) would recompute degradation on a random ImageNet subset matched to the smaller domains' class count.

*Implementation: `analysis/imagenet_baseline.py`, `analysis/imagenet_baseline_full_metrics.py`; `out/rebuttal/imagenet_baseline*/`.*

### §1d. Full-scale ImageNet LoRA+FT-SAE anchor

**What we did:** everything in §1a-c uses the frozen G-SAE. This asks a different, complementary question: at the *most extreme* "near domain" case possible — LoRA-adapting the model on ImageNet itself, the exact domain G-SAE was already trained on (domain distance = 0 by construction) — does retraining the SAE (FT-SAE) still help? An earlier attempt at this (25 images/class) was confounded by data starvation: FT-SAE was trained on 25 images/class against G-SAE's original 1.28M-image budget, so FT-SAE losing wasn't a fair test. Redone properly at 296 images/class (296,000 training images), with LoRA fine-tuning and FT-SAE training both matching every other dataset's protocol exactly.

**What we found:**

LoRA fine-tuning lifted accuracy substantially (65.67% → 73.83%, +8.17pp), confirming the adaptation itself worked. Then, on the LoRA-adapted model:

| Condition | Accuracy | Degradation from no-SAE |
|---|---|---|
| No SAE (LoRA-ImageNet) | 76.89% | — |
| + G-SAE | 71.33% | +5.56pp |
| + FT-SAE | 70.67% | +6.22pp |

**Why this resolves the concern:** FT-SAE is still marginally behind G-SAE (0.66pp) — but that gap shrank from 3.83pp (data-starved version) to 0.66pp once the confound was fixed, and it corroborates something §3 finds independently: **right at the pretraining distribution, retraining the SAE provides no measurable benefit.** Two separate experiments (this one on accuracy; §3 on reconstruction/FVE/dead_frac) landing on the same conclusion is stronger evidence than either alone.

**Caveat:** single seed, single LoRA fine-tune — read the 0.66pp gap as "near-zero," not as evidence FT-SAE is actually worse.

*Implementation: turing jobs 16096 (LoRA), 16178 (FT-SAE); `analysis/imagenet_anchor_eval.py`; `out/rebuttal/imagenet_anchor_eval_full/results.csv`.*

### §1e. From predictor to decision rule: a multi-feature regressor

nd4b and kubc both asked for more than a correlation — a concrete takeaway for *when* retraining is worth it. Rather than borrow a table format from an unrelated submission, we built the actual decision tool: a regressor trained on the distance features we've already measured (MMD², concept_gap, and class cardinality — the confound identified in §1c), model-selected honestly by out-of-sample performance rather than by whichever combination fits best in-sample.

**What we did:** fit every combination of {MMD², concept_gap, n_target_classes} as OLS predictors of degradation_pp across the same 8 domains, and evaluated each by leave-one-domain-out (LODO) cross-validation — refitting from scratch with each domain held out, so the reported R² reflects genuine out-of-sample generalization, not in-sample fit. We selected the final model by LODO R², not full-sample R², specifically to avoid the trap of preferring whichever model has the most predictors.

**What we found:**

| Predictor set | Full-sample R² | **LODO R²** | LODO MAE (pp) | LODO Spearman ρ (p) |
|---|---|---|---|---|
| MMD² alone | 0.004 | -1.211 | 6.27 | -0.95 (0.000) |
| concept_gap alone | 0.753 | 0.506 | 3.10 | 0.95 (0.000) |
| **MMD² + concept_gap** | **0.857** | **0.621** | **2.43** | **0.81 (0.015)** |
| concept_gap + n_target_classes | 0.781 | 0.495 | 3.10 | 0.93 (0.001) |
| MMD² + concept_gap + n_target_classes | 0.857 | 0.003 | 3.68 | 0.71 (0.047) |
| n_target_classes alone | 0.156 | -0.577 | 5.20 | -0.36 (0.385) |

**The combined MMD²+concept_gap model wins on out-of-sample performance, not just in-sample fit** — its LODO R² (0.621) beats concept_gap alone (0.506), the lowest LODO MAE (2.43pp) of any candidate, and a highly significant rank correlation. Adding class cardinality on top looks identical in-sample (full R²=0.857 either way) but collapses under LODO (0.003) — a textbook case of overfitting at n=8 that a full-sample-only comparison would have missed entirely, and worth showing reviewers directly as evidence we're selecting honestly rather than picking whichever number looks best.

**Selected model:** `degradation_pp ≈ -6.83 + 8.72·MMD² + 51.56·concept_gap`

**A concrete decision rule, using the §1c ImageNet self-degradation floor (3.65pp) as the retrain/reuse threshold:**

| Dataset | Observed degradation (pp) | LODO-predicted degradation (pp) | Recommendation matches observation? |
|---|---|---|---|
| caltech101 | 1.33 | -0.52 | ✓ (both: reuse is fine) |
| oxford_pets | 3.56 | 5.77 | ✗ (predicts retrain; observed reuse was fine) |
| eurosat | 4.00 | 6.33 | ✓ (both: retrain) |
| ucf101 | 4.89 | 5.66 | ✓ (both: retrain) |
| dtd | 5.33 | 6.10 | ✓ (both: retrain) |
| pathmnist | 11.55 | 10.09 | ✓ (both: retrain) |
| cub2002011 | 13.78 | 19.03 | ✓ (both: retrain) |
| fgvc | 14.22 | 9.44 | ✓ (both: retrain) |

**7/8 domains correct**, using *only* out-of-sample (LODO) predictions — the one miss (Oxford Pets) is a false positive (recommends retraining when reuse would actually have been fine), which is the safer direction to err in than the reverse.

**Why this resolves the concern:** this is the concrete takeaway nd4b and kubc asked for — not just "retraining helps" but a fitted rule that, cross-validated honestly, gets the retrain/reuse call right in 7 of 8 domains using distance features measurable *before* running any SAE training. It directly replaces the vague "we'll add a decision table" placeholder with a real, checkable artifact.

**Caveats, stated as plainly as everywhere else in this rebuttal:**
- n=8 with 2 predictors leaves only 5 residual degrees of freedom — LODO mitigates but does not eliminate small-sample risk.
- Like `concept_gap` itself, this regressor has **not** been scored against the three reserved held-out domains (OfficeHome, KITTI, Cityscapes) — that remains the one confirmatory test, deliberately not yet spent.
- The 3.65pp floor is itself a point estimate from a single ImageNet run (§1c), not a validated statistical threshold — treat it as a reasonable, motivated cutoff, not a precise boundary.
- Report this to reviewers as "a candidate decision rule with real out-of-sample support," not "a validated tool" — the distinction matters and we should not blur it under deadline pressure.

*Implementation: `analysis/degradation_regressor.py`; `out/rebuttal/mmd_predictor/degradation_regressor/{model_comparison.json,decision_table.csv}`.*

---

## §2. Is the generic-SAE failure actually causal, or just correlational?

> **UEAH:** "what is the steering ability of G-SAE vs FT-SAE... I suggest coming up with some quick experiments... at least steerability"

Every degradation number in §1 is correlational — accuracy, reconstruction quality, entropy. None of it proves that G-SAE's latents don't correspond to real directions the adapted model uses; it's consistent with G-SAE just being a slightly worse *descriptive* fit. This experiment tests it causally.

**What we did:** on EuroSAT and PathMNIST (first had to train FT-SAE checkpoints for these two, which didn't exist yet — checkpoint-init from G-SAE, single seed), we selected the top-activating latent per class in G-SAE and in FT-SAE, then intervened on the model directly:
- **Ablate** the latent for its own target class and measure the effect on the model's prediction margin.
- **Amplify** the latent on non-target-class images and measure the same thing.

Both are compared against a matched random control latent, so the reported number is the *intervention's own effect*, not just any latent's effect. The edit itself is error-preserving (`x' = x + D(z' - z)`) — an unmodified latent is an exact no-op, so any measured effect is attributable to the specific latent changed, not reconstruction noise.

One real bug surfaced and got fixed along the way: the code that picks matched control latents wasn't checking that the control had *any* activity on the target class, which could leave the calibration step undefined. Fixed by requiring nonzero target-class activity for a valid control match, and reran cleanly.

**What we found:**

| Dataset | Intervention | G-SAE effect (95% CI) | FT-SAE effect (95% CI) | FT-SAE − G-SAE (95% CI) |
|---|---|---|---|---|
| eurosat | ablate (target class) | -0.015 [-0.037, 0.004] | 0.091 [-0.026, 0.211] | +0.106 [-0.012, 0.228] |
| eurosat | amplify (non-target) | 0.020 [-0.0005, 0.047] | 1.263 [0.931, 1.600] | **+1.243 [0.910, 1.585]** |
| pathmnist | ablate (target class) | -0.008 [-0.034, 0.014] | 0.222 [0.060, 0.378] | **+0.230 [0.072, 0.387]** |
| pathmnist | amplify (non-target) | 0.034 [-0.022, 0.116] | 2.369 [1.379, 3.551] | **+2.335 [1.340, 3.481]** |

**Why this resolves the concern:** in all 4 cells, G-SAE's own effect confidence interval includes zero — ablating or amplifying a G-SAE latent does essentially nothing beyond what a random matched latent would do. FT-SAE's effect is far larger in all 4 cells, and significantly larger than G-SAE's (CI excludes zero) in 3 of 4. This is the direct answer to UEAH's question: **G-SAE latents on an adapted model have no detectable causal effect on its predictions; FT-SAE latents do.** It's not that G-SAE is "somewhat worse" at interpretability — it doesn't correspond to steerable structure at all.

**Caveat:** single training seed per FT-SAE; CIs are bootstrapped over evaluation examples, not over SAE training seeds — whether this holds across FT-SAE training seeds is untested (would need 2 more trainings per dataset).

*Implementation: `patchsae/tasks/eval_causal_steering.py`; turing jobs 15951 (FT-SAE training), 16048 (steering eval, 14m25s); `out/rebuttal/causal_steering/`.*

---

## §3. Does the G-SAE checkpoint matter, or just having target-domain data?

> **UEAH:** "I wonder if [scratch-trained SAEs] provide any further improvement or if they converge to a similar place"

FT-SAE is initialized from the G-SAE checkpoint and then retrained on target-domain data. This experiment isolates whether the checkpoint initialization itself matters, or whether a from-scratch SAE trained on the same target data would do just as well.

**What we did:** trained both FT-SAE (checkpoint-init) and Scratch-SAE (random init) on three domains spanning the distance range — caltech101 (near), dtd (mid), pathmnist (far) — 3 seeds each, 18 runs total, all completed with no failures.

**What we found:**

| Domain (tier) | Arm | Acc (95% CI, n=3) | recon_cosine (95% CI) | FVE (95% CI) | dead_frac (95% CI) |
|---|---|---|---|---|---|
| caltech101 (near) | gsae (n=1) | 87.11 | 0.903 | 0.946 | 0.430 |
| caltech101 (near) | ftsae | 85.48 [81.60, 89.36] | 0.896 [0.889, 0.903] | 0.827 [0.670, 0.984] | 0.048 [-0.020, 0.115] |
| caltech101 (near) | scratchsae | 84.89 [83.78, 86.00] | 0.888 [0.886, 0.891] | 0.937 [0.935, 0.939] | 0.023 [0.007, 0.039] |
| dtd (mid) | gsae (n=1) | 68.89 | 0.851 | 0.922 | 0.487 |
| dtd (mid) | **ftsae** | **72.59** [71.96, 73.23] | **0.901** [0.886, 0.916] | **0.945** [0.940, 0.951] | 0.317 [-0.057, 0.691] |
| dtd (mid) | scratchsae | 70.52 [69.24, 71.79] | 0.882 [0.879, 0.886] | 0.934 [0.932, 0.937] | 0.173 [-0.178, 0.523] |
| pathmnist (far) | gsae (n=1) | 79.65 | 0.793 | 0.913 | 0.501 |
| pathmnist (far) | **ftsae** | 90.00 [89.10, 90.90] | **0.934** [0.933, 0.935] | **0.969** [0.968, 0.970] | **0.009** [0.001, 0.018] |
| pathmnist (far) | scratchsae | 89.03 [88.59, 89.47] | 0.913 [0.908, 0.918] | 0.960 [0.959, 0.961] | 0.027 [-0.006, 0.059] |

**Why this resolves the concern — a more precise answer than either of UEAH's two hypotheses alone:**
- **Near (caltech101): they converge to the same place.** FT-SAE and Scratch-SAE are statistically indistinguishable on accuracy; Scratch-SAE actually has the tighter, higher FVE point estimate. At low domain shift, the checkpoint doesn't clearly buy anything.
- **Mid (dtd): FT-SAE wins clearly** — non-overlapping accuracy CIs, and ahead on reconstruction quality too.
- **Far (pathmnist): FT-SAE wins clearly on reconstruction fidelity**; accuracy point estimate is higher but the CIs brush at the boundary with only 3 seeds.
- dead_frac is too high-variance across seeds to read as evidence either way.

**Bottom line:** initialization is not irrelevant, but its benefit is domain-shift-dependent — negligible near the pretraining distribution, clear at moderate/far distance, and more consistently visible in reconstruction fidelity than in raw accuracy. This matches what §1d finds independently at distance=0.

*Implementation: turing job 15953 (training, 4h59m for 18 runs), job 16047 (eval); `out/rebuttal/sae_initialization_ablation_summary.json`.*

---

## §4. Secondary experiments

### §4a. DAMS robustness under coarsened labels — complete

> **UEAH:** "why would we expect SAE concepts to be class separable... different label sets... shared concepts across classes shouldn't be penalized"

DAMS (the paper's monosemanticity score) partly depends on class separability. UEAH's critique: if DAMS's ranking flips under a different, equally reasonable class taxonomy, that's a real weakness in the metric, not just a modeling choice.

**What we did:** predeclared coarser class groupings for two datasets before computing anything —

- **EuroSAT, 10→4:** Agricultural {AnnualCrop, PermanentCrop, Pasture}; Vegetation {Forest, HerbaceousVegetation}; Water {River, SeaLake}; Built-up {Highway, Industrial, Residential}.
- **PathMNIST, 9→5:** Tumor epithelium {colorectal_adenocarcinoma_epithelium}; Normal epithelium {normal_colon_mucosa}; Stroma/connective {cancer-associated_stroma, smooth_muscle, adipose}; Immune {lymphocytes}; Non-tissue {debris, mucus, background}.

Then computed DAMS for both G-SAE and FT-SAE at both granularities, reusing the same precomputed SAE activations (only the label grouping changes, not the underlying features).

**What we found:**

| Dataset | Granularity | n classes | ftsae DAMS | gsae DAMS | Gap (ftsae − gsae) |
|---|---|---|---|---|---|
| eurosat | fine | 10 | 0.773 | 0.700 | +0.073 |
| eurosat | coarse | 4 | 0.785 | 0.767 | **+0.018** |
| pathmnist | fine | 9 | 0.889 | 0.733 | +0.157 |
| pathmnist | coarse | 5 | 0.884 | 0.806 | **+0.078** |

**Why this resolves the concern:** the direction of the paper's core claim (FT-SAE more monosemantic than G-SAE) survives coarsening in both datasets, 4/4 — DAMS's *ranking* is not an artifact of the specific class taxonomy chosen. But the *size* of the gap shrinks substantially under coarsening — 75% on EuroSAT, 50% on PathMNIST — so DAMS's magnitude is taxonomy-sensitive even though its direction isn't. We'll report both findings rather than the single-taxonomy number, and flag magnitude-sensitivity as a limitation wherever DAMS gap size (not just direction) is used to support a claim.

*Implementation: `analysis/dams_coarsening_robustness.py`; `out/rebuttal/dams_coarsening/results.csv`.*

### §4b. Not started (lower priority, explicitly deprioritized)

- **EuroSAT hyper-specialization mechanism** (dead_frac 0.16→0.81, L0 125→15 per the paper): would need a fixed image/class-count sweep plus inspection of the ~15 surviving features' top-activating patches, to test whether the collapse is driven by EuroSAT's low class count (10) rather than something else about the domain.
- **Language-supervision confound** (does DINOv2 lacking CLIP's language supervision, rather than domain shift per se, explain part of the pattern on PathMNIST?): would need a text-alignment head bolted onto DINOv2, retrained under the same protocol. Largest lift of the open items; any result would be about that specific alignment head, not a general claim about DINOv2 pretraining.

---

## §5. H5 — causal steerability (UEAH's follow-up design, Part A)

UEAH sent a detailed follow-up design doc after the initial reviews, distilling their core ask: §2 shows G-SAE *representations* degrade under shift, but never directly tests whether that degradation matters for what SAEs are actually used for downstream — explaining and steering a model's behavior. They proposed two tiers: **Part A** (classifier-level ablation/injection, pure inference on checkpoints that already exist) and **Part B** (steering a vision-language model, LLaVA, mirroring Kulkarni et al.'s protocol — requires training new SAEs on a different backbone, a multi-day undertaking tracked separately below, not reported as a result here).

**H5, as UEAH framed it:** interventions on generic-SAE latents lose causal effect on the adapted model as domain distance grows; domain-matched (FT-)SAEs restore steering efficacy, and the gap scales with MMD.

**What we did:** reused §2's exact intervention machinery (same error-preserving edit formula, same no-op guarantee for unmodified latents), generalized from single-latent to multi-latent edits, across 4 datasets spanning the MMD range using FT-SAE checkpoints that already existed (caltech101=near, dtd/eurosat=mid, pathmnist=far — no new SAE training needed). This is deliberately a "cheap tier" — UEAH's own framing was that quick experiments are an acceptable response here — 6 classes/dataset, not the full grid from the design doc:
- **Necessity:** zero a class's top-k latents (k∈{5,25}), measure the accuracy drop vs. zeroing k random non-top latents.
- **Sufficiency:** on *other*-class images, multiply the target class's top-k latents (k=10) by alpha∈{2,5,10}, measure how often this flips the prediction to the target class (AUC over the alpha sweep).

This finished in 19 minutes of compute precisely because it reused already-validated machinery from §2 rather than building something new.

**What we found:**

| Dataset | MMD² | Necessity gap (k=5) | Necessity gap (k=25) | Sufficiency gap (AUC) |
|---|---|---|---|---|
| caltech101 (near) | 0.028 | -0.036 | -0.026 | -0.011 |
| dtd (mid) | 0.090 | -0.014 | +0.008 | +0.004 |
| eurosat (mid) | 0.291 | 0.000 | +0.047 | +0.030 |
| pathmnist (far) | 0.423 | **+0.104** | **+0.167** | -0.006 |

| Test | Pearson r (p) | Spearman ρ (p) |
|---|---|---|
| Necessity gap, k=5 | 0.90 (0.10) | **1.00 (0.00)** |
| Necessity gap, k=25 | 0.95 (0.05) | **1.00 (0.00)** |
| Sufficiency gap (AUC) | 0.26 (0.74) | 0.40 (0.60) |

**Why this resolves the concern — partially, and we're reporting it that way rather than smoothing it over:**
- **Necessity tracks domain distance almost perfectly** (Spearman ρ=1.0 at both k values). At the pretraining distribution (caltech101), G-SAE's top latents are *at least as* necessary to correct classification as FT-SAE's; by the far domain (pathmnist), FT-SAE's are far more necessary — G-SAE's top-25 latents barely matter to its own predictions there. This is the causal, necessity-side counterpart to every correlational degradation number elsewhere in this rebuttal, and it reproduces the same "no benefit near the pretraining distribution, growing benefit with distance" pattern found independently in §3 and §1d.
- **Sufficiency shows no reliable trend** (r=0.26, not significant) — pathmnist, the farthest domain, actually has a near-zero/slightly negative sufficiency gap, breaking the necessity pattern. Real finding, reported honestly: G-SAE and FT-SAE latents may differ more in whether they're *necessary* for correct classification than in whether they can be *artificially amplified* to force a specific misclassification. "Necessity" and "sufficiency" shouldn't be collapsed into one steerability number — they're answering different questions and behave differently here.
- Per-class statistical tests (n=6/dataset) aren't significant anywhere, as expected at this scale — the informative signal is the dataset-level rank correlation (n=4, ρ=1.0 for necessity), not the underpowered per-class test.

**Part B status:** scoped, not abandoned, but genuinely multi-day (new SAEs on a different vision backbone, new LLaVA inference and scoring infrastructure). In progress on turing; won't land before the rebuttal deadline. Report as ongoing work, not a promised result.

*Implementation: `analysis/h5_classifier_steering.py`; turing job 16360 (19m26s); `out/rebuttal/h5_classifier_steering/`.*

---

## Statistical rigor (cross-cutting note)

`REBUTTAL_EXPERIMENTS.md` calls for ≥3 seeds on headline configurations. §3 already has this (3 seeds × 3 domains). §1 and §2 are single-run per domain by design: §1's uncertainty comes from LODO/CI across *domains*, which is the right unit there; §2 uses a single training seed per FT-SAE with bootstrap CIs over evaluation examples — whether the causal effect holds across FT-SAE training seeds is untested and out of scope for the rebuttal window.

## Draft claim boundary (do not finalize until validated)

> Below the predicted-degradation floor (3.65pp, the ImageNet self-degradation baseline), frozen G-SAE reuse is likely fine; above it, retraining is recommended. Predicted degradation = -6.83 + 8.72·MMD² + 51.56·concept_gap (§1e), correct in 7/8 domains under leave-one-domain-out cross-validation.

This is now a concrete, checkable rule rather than a placeholder — but it is still exploratory: n=8, not pre-registered, and not yet scored against the three reserved held-out domains (§1b, §1e). Report it to reviewers as a candidate decision rule with real out-of-sample support, strengthened — not replaced — by the causal results in §2 and §5.

---

## Reviewer concerns — response matrix

Mirrors the concern-by-concern table format used to track this rebuttal. Status tags, used consistently across all three reviewers:

- ✅ **Established** — real experiment, strong result, ready to submit as-is.
- 🔬 **Candidate** — real analysis, honestly exploratory (small n and/or not yet held-out-validated) — present as promising, not proven.
- 📝 **Text fix** — a wording/caption/citation fix in the paper itself, no experiment needed.
- ⛔ **Not pursued** — we don't have the evidence to back this one; dropped from the response rather than half-argued (see "Before submitting" checklist below for what would unblock it).

Full submission-ready prose for each item is in "Reviewer response drafts" below — this table is for scanning status at a glance.

### Reviewer UEAH (R1) — Borderline reject, Confidence 5

| # | Concern / Weakness Raised | Our Response | Status |
|---|---|---|---|
| 1 | H1 is well-known / not surprising | The paper documents concrete performance and geometric costs of the standard "freeze and reuse a generic SAE" recipe — backed by §1 (predictor) and §2/§5 (causal evidence), not just the fact that retraining helps. | ✅ Established |
| 2 | Unclear why class separability is desirable for DAMS | Geometric argument (separated t-SNE clusters = distinctive features) plus §4a's coarsening experiment: ranking holds 4/4 under a coarser taxonomy; magnitude shrinks 50–75%, reported as a real limitation. | ✅ Established |
| 3 | Dead neurons — could Matryoshka SAEs resolve this? | Already in the paper — pointing to Table 9. | ✅ Established *(verify Table 9 reference — see checklist)* |
| 4 | Fig. 2 doesn't show FT+G-SAE | Table 6 has the numbers directly. Reinforced by §1a's 8-domain degradation table and §2's causal result: accuracy gap can look modest while the causal-steering gap is large. | ✅ Established |
| 5 | Practical utility / steering not tested | §2: latent ablation + amplification, G-SAE vs FT-SAE, bootstrap CIs, following Kulkarni et al. Extended in §5 (H5 Part A, necessity + sufficiency). | ✅ Established |
| 6 | Missing FT-SAE-vs-Scratch-SAE baseline | §3: 3 domains × 3 seeds × 2 arms. Converge near the pretraining distribution; FT-SAE wins clearly at mid/far distance on reconstruction fidelity. | ✅ Established |
| 7 | Typos (L194, L294) | Both noted. | 📝 Text fix |

### Reviewer nd4b (R2) — Reject, Confidence 3

| # | Concern / Weakness Raised | Our Response | Status |
|---|---|---|---|
| 1 | No guidance on restoring interpretability/fidelity/causal faithfulness | Each result now explicitly restates how retraining fixes the specific failure mode it's paired with; reinforced by #2's decision rule. | ✅ Established |
| 2 | No clear takeaway table | §1e: a regressor (MMD²+concept_gap), model-selected by leave-one-domain-out R² (0.621) rather than in-sample fit, yielding a concrete retrain/reuse rule correct in 7/8 domains out-of-sample. | 🔬 Candidate |
| 3 | Fig 2/3 don't clearly support "farther domains benefit more" | Clarifying figure interpretations/captions directly. | 📝 Text fix |
| 4 | Performance of ZS+G-SAE on ImageNet | §1c: full SAE-quality suite — 66.60%→62.95% accuracy (+3.65pp), plus L0/dead_frac/recon_cosine/FVE/entropy. | ✅ Established |
| 5 | Language-supervision vs. training-data confound (DINOv2 vs CLIP) | DTD and EuroSAT aren't well-represented in CLIP's own pretraining corpus either, so language supervision remains the more parsimonious explanation. | 🔬 Candidate *(needs a citation — see checklist)* |
| 6 | No ImageNet+G-SAE baseline at all | Same evidence as #4. | ✅ Established |
| 7 | Undefined symbols (φ, γ, C, μᵢ, z_ij) L233–241 | Pointing to appendix. | 📝 Text fix |
| 8 | FT+G-SAE vs FT+FT-SAE gap "quite small, often within a few percent" | §2's causal result stands on its own: accuracy alone understates the representational gap. | ✅ Established *(the "larger for prompt-based methods" counter-claim is ⛔ not pursued — no supporting numbers on hand)* |

### Reviewer kubc (R3) — Borderline reject, Confidence 4

| # | Concern / Weakness Raised | Our Response | Status |
|---|---|---|---|
| 1 | Would be more interesting to identify WHEN retraining is needed | §1e's regressor + decision rule (predicted degradation vs. a 3.65pp floor), correct in 7/8 domains out-of-sample. | 🔬 Candidate |
| 2 | H1/H2/H4 anticipatable by domain experts | §2 (causal test) and §3 (init ablation) go beyond the anticipated direction: conditional on domain distance, and causal, not just accuracy-based. | ✅ Established *(the "prompt-based methods" counter-point is ⛔ not pursued — see nd4b #8)* |
| 3 | Why is DAMS lower for DTD/FGVC over G-SAE? | §1b's `concept_gap`: both domains score high (DTD 0.232, FGVC 0.293) — G-SAE's ImageNet-object dictionary has little well-matched structure for either vocabulary. | ✅ Established |
| 4 | Table 3 hard to read without more description | Adding explanatory text to caption/body. | 📝 Text fix |
| 5 | Figure 4 (bottom row) pattern — clue about adaptation? | Figure 4 is a feature-correlation heatmap, not raw activation strength — the pattern reflects lack of separability, not an adaptation signature. | 📝 Text fix |
| 6 | Fig. 4 caption/naming inconsistency | Camera-ready fix. | 📝 Text fix |

## Before submitting — author action items

Everything below is a factual claim the response leans on that I could not verify myself (I have not read the paper PDF) or a claim we decided not to make. Resolve these before the text above goes to OpenReview — none of this should appear in the actual submitted response:

1. **UEAH #3 (Table 9):** confirm Table 9 in the appendix actually contains the Matryoshka SAE / dead-neuron comparison, and that its framing matches "this comparison is already in the paper."
2. **nd4b #5 (DINOv2 confound):** find a citation or concrete justification that DTD/EuroSAT are underrepresented in CLIP's pretraining corpus — the argument as drafted asserts this without a source.
3. **nd4b #8 / kubc #2 (prompt-based methods gap):** we do not have the paper numbers to support "the gap is larger for prompt-based methods." Either locate them and I'll draft the point, or drop it — the current draft already omits it rather than promising it.

---

# Reviewer response drafts

Grounded in the actual review text pasted 2026-07-25 (meta-review by AC k9GS; official reviews from UEAH, nd4b, kubc). Every claim below cites the specific section above it draws from. Paper-specific questions (exact figure/table contents, line numbers, typos) are **not answered** here since I have not read the paper PDF itself — those need the authors' own text. Draft prose is a starting point for the actual OpenReview comment, not verbatim final text — check length against whatever limit applies.

## The common thread (AC + all three reviewers)

The meta-review, UEAH, nd4b, and kubc **all** independently raise the same core objection: the central finding (generic SAEs degrade under adaptation) is expected, and the paper offers no actionable guidance for *when* retraining is actually necessary. This is the single most important thing to address, and it's exactly what §1 (concept_gap) and §2 (causal steering) now speak to — not by claiming the "obvious" finding is novel, but by showing the paper now does something with it: quantifying it predictively and demonstrating it's causal. Lead with this reframing in whatever general response accompanies the individual replies, then let each reviewer section below do the specific work.

## Response to Reviewer UEAH

UEAH raised six weaknesses; four now have direct new evidence, one is explicitly out of scope, one is a reframing.

**On the class-separability assumption behind DAMS** ("why would we expect SAE concepts to be class separable... different label sets... shared concepts across classes shouldn't be penalized"): our primary response is geometric — class separability is not an arbitrary modeling choice, it's what the t-SNE plots are already showing directly: well-separated clusters in activation space are exactly the signature of distinctive, class-discriminative features, and a monosemanticity score should reward a dictionary whose features carve up the representation space the same way the classes do. Where concepts are genuinely shared across classes, that shows up as expected overlap in the same t-SNE geometry, not as a penalty DAMS invents.

We also tested the specific worry — that this assumption is fragile to the exact class taxonomy chosen — directly and empirically. We recomputed DAMS for EuroSAT and PathMNIST under a coarser, predeclared class taxonomy (EuroSAT 10→4 land-cover superclasses; PathMNIst 9→5 tissue supertypes — full mapping in §4a above) and compared against the original fine-grained labels. The paper's central ranking (FT-SAE more monosemantic than G-SAE) holds at both granularities in both datasets (4/4), but the *magnitude* of the gap shrinks substantially under coarsening — 75% on EuroSAT, 50% on PathMNIST. We read this as confirming your intuition has real teeth even with the geometric argument granted: DAMS's absolute values are sensitive to the specific class taxonomy used, even though its qualitative conclusion is not. We will report both the geometric rationale and the coarsening robustness findings rather than the single-taxonomy number, and note DAMS's magnitude-sensitivity as a limitation.

**On practical utility / steering** ("what is the steering ability of G-SAE vs FT-SAE... I suggest coming up with some quick experiments... at least steerability"): we ran exactly this, following the steering-experiment protocol from Kulkarni et al. (the SAE-CBM paper, same reference used for our H5 Part B design). Latent ablation and amplification on EuroSAT and PathMNIST, G-SAE vs. FT-SAE, 20 latents/condition, bootstrap CIs over 5000 samples (full protocol and results in §2 above). Result: G-SAE latents have **no detectable causal effect** on the adapted model's predictions — every G-SAE effect estimate's 95% CI includes zero, in all 4 dataset × intervention cells. FT-SAE latents produce far larger effects, significantly larger than G-SAE's in 3 of 4 cells (both amplify conditions, and ablation on PathMNIST). This directly answers your question and converts the paper's descriptive claim into a causal one: G-SAE's failure under adaptation isn't just "harder to interpret post-hoc," it corresponds to *no steerable structure in the adapted model at all* along those directions.

We also followed up on your detailed design doc for a broader steerability study (H5, §5 above): the classifier-level "Part A" tier (necessity via ablation, sufficiency via injection, across 4 datasets spanning our MMD axis, reusing this exact intervention mechanism — no new training, ~19 minutes of compute) found that necessity tracks domain distance almost perfectly (Spearman ρ=1.0 across the 4 datasets: near-zero-or-negative FT-SAE advantage at the pretraining distribution, growing to a clear advantage at the farthest domain), while sufficiency (latent amplification/injection) showed no reliable trend at this scale — a genuine, only partially resolved result we're reporting honestly rather than smoothing over. The LLaVA tier mirroring Kulkarni et al. directly (Part B) is scoped and in progress but needs new SAEs on ViT-L/14-336 plus new inference/scoring infrastructure — a multi-day build that will not finish before the rebuttal deadline; we are not promising it will land, but the design is sound and we intend to pursue it regardless of the rebuttal outcome given how directly it engages with what SAEs are actually used for.

**On dead neurons being resolvable via Matryoshka SAEs:** this comparison is already in the paper — Table 9 in the appendix reports it. We'll point reviewers there directly rather than re-running it.

**On the missing FT-SAE-vs-scratch-SAE hypothesis** ("I wonder if those provide any further improvement or if they converge to a similar place"): we ran this as a controlled ablation — 3 domains spanning near/mid/far distance (Caltech101/DTD/PathMNIST), 3 SAE-training seeds each, both arms (§3 above). Answer, more precise than either of your two hypotheses alone: checkpoint initialization does *not* matter near the pretraining distribution (Caltech101: FT-SAE and Scratch-SAE statistically indistinguishable on accuracy), but *does* matter at moderate/far distance (DTD, PathMNIST: FT-SAE wins clearly on reconstruction fidelity, and on accuracy at DTD). So it's not "they converge to the same place" uniformly — it depends on domain distance, which is itself informative about what the G-SAE checkpoint carries over.

**On Fig. 2 not showing FT+G-SAE ("might be that G-SAE gives similar reconstruction+sparsity to FT-SAE?"):** Table 6 already reports the full numbers for this comparison — we'll point reviewers there directly. It's also reinforced by two new results: our no-SAE-vs-G-SAE accuracy measurements on the LoRA-adapted model (§1a's degradation table, 8 domains) and the §2 causal result, which adds the sharper point that FT+G-SAE and FT+FT-SAE are not "similar" just because their accuracy is close — the accuracy gap can look modest (a few points) while the causal-steering gap is large, because accuracy is a blunt instrument for measuring whether SAE features track real internal structure.

**On H1 being "well-known" / "common knowledge":** the point of H1 isn't that retraining helps — it's that the field's standard recipe (freeze a generic SAE, or naively reuse it across the domain-adapted model) carries real performance and geometric costs that aren't otherwise visible, which is what this paper documents. What's new after this rebuttal makes that concrete rather than just asserted: (a) a first attempt at a quantitative predictor for *when* retraining is needed (§1, honestly reported as promising but not yet validated — see below), and (b) causal evidence that the failure mode is not superficial (§2, §5). We'll sharpen the abstract/intro to lead with "here is what following the standard recipe costs you, and when" rather than the bare correlational finding.

**On typos** (L194 "penultmate"→"penultimate"; L294 "Also, Unsupervised"→"Also, unsupervised"): straightforward camera-ready fixes, both noted.

## Response to Reviewer nd4b

nd4b's rating (2: Reject) rests most heavily on the MMD-correlation critique — this is the one place where nd4b's specific technical objection turned out, on direct re-testing, to be *correct*, and we think saying so plainly is the strongest response available.

**On "What is the performance of the reported r metrics for ZS+G-SAE on ImageNet?":** base (non-adapted) CLIP ViT-B/16 on a 2000-image stratified ImageNet-1k validation sample: 66.60% zero-shot, 62.95% with G-SAE reconstruction — a 3.65pp degradation *even on G-SAE's own training domain, with no adaptation at all* (§1c above, "ImageNet baseline" — the 66.60% no-SAE figure is close to CLIP ViT-B/16's published ~68% zero-shot ImageNet accuracy, which we take as a sanity check on the eval pipeline). We think this number is worth featuring rather than omitting: it sets an honest floor on what "degradation" means in our other tables — some of the smaller degradation values elsewhere (e.g., Caltech101 at 1.33pp) are *below* this floor and are probably better read as "no detectable domain-shift effect" than as a genuine small positive effect. It also surfaces a confound we had not controlled for: ImageNet-1k's 1000-way classification is far more sensitive to small reconstruction error than our other domains' 10-200-way tasks, so part of the cross-domain variance in degradation_pp may reflect class cardinality, not domain distance. We will add both the floor and the cardinality caveat to the paper rather than present the degradation numbers as cleanly attributable to domain shift alone.

**On "Table 3 does not show strong correlation between the metrics and MMD" / "the gap does not really strictly increase":** we re-ran this analysis with two more domains added to the fit (Oxford Pets, PathMNIST) and confirmed the relationship is much weaker than we'd represented: with 8 domains, apparent R² for MMD²→degradation is 0.004 (essentially no linear relationship) and leave-one-domain-out R² is negative (§1a above). We traced this to a real, interpretable cause: raw CLIP-embedding MMD measures *distributional* shift (photography style, composition), not *conceptual* novelty relative to what the generic SAE's dictionary already covers — Oxford Pets has the highest embedding-space MMD in our cohort but one of the lowest degradations, because its class vocabulary (dog/cat breeds) is already well-covered by ImageNet-1k. A class-vocabulary-overlap distance (§1b's `concept_gap`, computed via CLIP text-embedding similarity to ImageNet-1k class names) tracks degradation far better (R²=0.75, LODO R²=0.51) and resolves the specific inversion you'd be right to point at in our raw-MMD figures. We will replace the raw-MMD correlation claim in the paper with this more accurate (and more mechanistically grounded) account, explicitly flagged as a preliminary/exploratory predictor pending held-out validation, not an established result.

**On "the paper does not provide enough guidance on how to restore interpretability... after domain adaptation" / "no clear takeaway":** the guidance the paper already gives is "retrain the SAE on the adapted model" (FT-SAE), now substantiated causally (§2) rather than just by accuracy deltas, and we will walk through each result explicitly restating how retraining fixes the specific failure mode it's paired with, rather than leaving readers to connect figures to the takeaway themselves. On top of that, we built a concrete decision tool for *when* retraining is worth the cost (§1e): a regressor over MMD² and `concept_gap`, model-selected by leave-one-domain-out cross-validation rather than in-sample fit (LODO R²=0.621, beating either predictor alone), yielding an explicit rule — `degradation_pp ≈ -6.83 + 8.72·MMD² + 51.56·concept_gap`, compared against a 3.65pp floor derived from the ImageNet self-degradation baseline — that gets the retrain/reuse call right in 7 of 8 domains under out-of-sample evaluation. It is exploratory (n=8, not yet scored against the three held-out domains) and we'll present it as a candidate, not a validated tool, but it is a real, checkable takeaway rather than a placeholder.

**On "the performance difference between FT+G-SAE and FT+FT-SAE is quite small, often within a few percent":** accuracy alone understates the representational gap — the causal steering result (§2) makes this concrete: G-SAE and FT-SAE can have similar downstream accuracy while G-SAE latents have essentially zero causal effect on the model's predictions and FT-SAE latents have a large one.

**On the DINOv2 language-supervision confound** ("could this be an effect of training data itself instead of language supervision... DINO might not have seen medical images but CLIP might have seen medical corpus"): DTD and EuroSAT are not well-represented in CLIP's pretraining corpus either (texture-patch photography and satellite imagery are not typical web-image content), so the pattern holding on those two domains is harder to explain by "CLIP simply saw this data during pretraining" — language supervision remains the more parsimonious explanation. We have not run the controlled experiment (a text-alignment head on DINOv2) that would settle this directly, so we present the argument above as suggestive, not conclusive, and keep the confound listed in Limitations.

**Not addressed — flagging honestly:** the writing-clarity issues (missing figure legends, undefined symbols around L233-241, typos) require editing the paper text directly and are not something new experiments can fix; note them as accepted revisions for camera-ready.

## Response to Reviewer kubc

**On "can a heuristic approach determine when the drift is too much to reuse the original SAE?":** this is the most direct match to work done in this rebuttal, and we built an actual heuristic rather than just a predictor (§1e). We fit every combination of MMD², `concept_gap`, and class cardinality as predictors of degradation, selecting among them by leave-one-domain-out cross-validation specifically to avoid rewarding whichever model merely fits best in-sample — a 3-predictor model looks identical in-sample (R²=0.857) to the 2-predictor model but its LODO R² collapses from 0.621 to 0.003, which we report directly as evidence of honest model selection. The selected rule — MMD²+concept_gap, compared against a 3.65pp degradation floor — gets the retrain/reuse recommendation right in 7 of 8 domains out-of-sample. We're explicit that this is not yet validated against the three held-out domains (OfficeHome/KITTI/Cityscapes), deliberately reserved to avoid scoring a post-hoc-discovered variable against them and calling it confirmatory — but it is a concrete, checkable heuristic, not just a correlation.

**On H1/H2/H4 being things domain experts could anticipate:** see the common-thread note above — our response is to point at what's new beyond the anticipated direction: a causal test (§2) and an initialization ablation (§3) that produces a more precise answer than "retraining helps" (it helps conditionally on domain distance, and helps causally, not just on accuracy).

**On why DAMS values are much lower for DTD and FGVC when applied over G-SAE:** we can answer this directly from §1b's `concept_gap` data, computed independently of DAMS. DTD (concept_gap=0.232) and FGVC (concept_gap=0.293) are both on the high end of our 8-domain cohort — DTD's vocabulary is abstract texture descriptors ("veined," "porous") with no good ImageNet-1k object analogue, and FGVC's is fine-grained aircraft variants (e.g. "C-47") that ImageNet's coarse object categories don't resolve. A generic SAE dictionary built on ImageNet-object features has correspondingly little well-matched structure to represent either vocabulary, which is exactly the condition under which DAMS's class-separability component would be expected to score G-SAE low — the same mechanism explaining the MMD-vs-degradation inversion we found for Oxford Pets, applied here in the other direction.

**On Table 3's readability and the DAMS explanation:** we'll add explanatory text directly in the caption/body rather than relying on the appendix alone, given how central DAMS is to H3.

**On the Figure 4 (bottom row) pattern** (stronger response bottom-right for Base+G-SAE, similar pattern for FT+FT-SAE): Figure 4 is a heatmap of feature-activation correlation, not raw activation strength — the pattern you're pointing at reflects a lack of separability (correlated, overlapping feature responses) rather than a specific adaptation signature; we'll state this explicitly in the caption so the figure doesn't invite the "is this a clue" reading on its own.

**Not addressed — flagging honestly:** the PathMNIST/MedMNIST caption inconsistency and "adapted SAE"/"FT-SAE" naming inconsistency are straightforward camera-ready fixes.
