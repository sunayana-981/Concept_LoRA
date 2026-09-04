# Causal latent steering experiment

`eval_causal_steering.py` is the bounded intervention study used to compare
G-SAE and FT-SAE on LoRA-adapted CLIP. Its defaults are the intended primary
rebuttal experiment:

- datasets: EuroSAT and PathMNIST;
- adapted model: the dataset's fixed LoRA CLIP checkpoint;
- dictionaries: G-SAE and FT-SAE at the same `-2/resid` hook;
- 20 selected latents per dictionary and 20 matched random-latent controls;
- 256 training/selection images per class;
- 64 disjoint evaluation images per class;
- CLS-token intervention only.

An **FT-SAE** in this experiment means a dictionary initialized from G-SAE,
then fine-tuned on target-domain adapted-model activations. A legacy
random-initialized target-domain SAE is a **Scratch-SAE**, not an FT-SAE, and
must not be placed under the `ftsae` key in `rebuttal_sae_paths.json`.

## Run

From `patchsae/`, first perform the no-model preflight:

```bash
PYTHONPATH=. python tasks/eval_causal_steering.py --dry_run
```

The preflight resolves the dataset, LoRA, G-SAE, and FT-SAE paths and writes
`out/rebuttal/causal_steering/preflight.json`. The full run fails before
loading a model if either side of the comparison is missing.

After the new G-SAE-initialized FT-SAE checkpoints are registered:

```bash
PYTHONPATH=. python tasks/eval_causal_steering.py \
  --sae_paths configs/rebuttal_sae_paths.json \
  --device cuda
```

For a non-paper smoke test, use a separate output directory and a batch cap:

```bash
PYTHONPATH=. python tasks/eval_causal_steering.py \
  --datasets eurosat \
  --num_latents 10 \
  --selection_images_per_class 16 \
  --eval_images_per_class 8 \
  --debug_max_batches 2 \
  --out_dir out/debug/causal_steering
```

Never merge results produced with `--debug_max_batches` into paper tables.
`--allow_missing_conditions` is for debugging only: the main comparison
requires both G-SAE and FT-SAE.

## Fixed selection and controls

Selection uses no evaluation activations, labels, logits, or predictions.
Within the training/selection split, the script computes each latent's
one-vs-rest standardized class activation difference. It filters dead,
extremely rare, and near-universal latents. A seeded per-class quota distributes
the requested count as evenly as possible, and the strongest eligible unique
features fill each quota. The quota seed depends on the dataset, not the SAE,
so G-SAE and FT-SAE have the same target-class composition. Their actual
features are selected independently within their respective dictionaries.

Each selected latent gets one unique random control. Controls are sampled,
with a fixed seed, from the 50 closest unmatched active latents using:

- global activation magnitude and prevalence;
- target-class activation magnitude and prevalence;
- decoder-vector norm.

This matching prevents a selected-vs-random contrast from merely measuring
larger intervention magnitude.

## Intervention

For the natural feature vector `z = E(x)` and reconstruction `D(z)`, the
script retains the reconstruction error:

```text
error = x - D(z)
x_intervened = D(z_intervened) + error
              = x + D(z_intervened - z)
```

Only one latent changes:

- **ablate:** `z_j <- 0`;
- **amplify:** `z_j <- max(z_j, q_j)` by default, where `q_j` is the 90th
  percentile of that feature's positive target-class activation on the
  selection split.

The calibrated quantile avoids comparing raw clamp values across SAEs with
different latent scales. `--amplify_mode multiply --amplify_factor 2` is a
pre-specified robustness variant.

The unedited reconstruction error means a no-change edit returns the original
activation exactly. The script also runs `D(z)` without this error term as a
separate SAE-reconstruction control. It never calls `forward_clamp`, never
sets the other 49,151 latents, and never subtracts a hard-coded bias.

CLIP image and text embeddings are explicitly L2-normalized before logits and
probabilities are computed.

All causal effects in `effects.csv`, `paired_effects.csv`, and the primary
aggregate use the raw adapted model with no hook (`x`) as their paired
reference. They do **not** use `D(z)` as the baseline. The latter is reported
only in `reconstruction_controls.csv` to expose reconstruction confounding.

## Primary analysis

The signed primary endpoint is target-class logit-margin movement:

- ablation, evaluated on target-class images:
  `-(intervened margin - baseline margin)`;
- amplification, evaluated on non-target images:
  `intervened margin - baseline margin`.

Positive values therefore mean steering in the hypothesized direction. For
each selected latent, subtract its matched control's signed effect. Aggregate
the 20 paired differences with a latent-level bootstrap confidence interval.
The direct comparison is:

```text
FT-SAE mean control-adjusted effect - G-SAE mean control-adjusted effect
```

This is an **arm-level contrast between independently selected latent sets**,
not a feature-paired G-SAE/FT-SAE test. “Paired” elsewhere in the outputs means
selected latent versus its matched random control within one dictionary.

Accuracy changes, target prediction rates, probabilities, flip rates, and
the unadjusted selected/control effects are secondary diagnostics. Report
both amplification and ablation rather than choosing whichever is larger.

## Outputs

All outputs are written under `--out_dir`:

| File | Unit of observation | Purpose |
|---|---|---|
| `preflight.json` | run | Exact resolved inputs and missing requirements |
| `run_metadata.json` | run/cell | Arguments, protocol version, completed and skipped cells |
| `latent_manifest.csv` | latent or matched control | Frozen selection score, target class, matching diagnostics, calibrated intervention, checkpoint |
| `reconstruction_controls.csv` | dataset × SAE | Accuracy/agreement/KL change from full SAE reconstruction |
| `effects.csv` | latent × intervention × eval group | Aggregate accuracy, target-rate, logit, probability, margin, and flip effects |
| `paired_effects.csv` | selected-control pair × intervention × group | Within-pair selected-minus-control effects |
| `aggregate_results.csv` | dataset × SAE × intervention | Primary signed, control-adjusted means and latent-bootstrap 95% CIs |
| `condition_contrasts.csv` | dataset × intervention | FT-SAE minus G-SAE primary contrast |
| `per_example_effects.csv.gz` | image × latent × intervention | Paired raw effects for alternative tests and plots |

The three evaluation groups in `effects.csv` are `target`, `non_target`, and
`all`. The primary group is fixed to `target` for ablation and `non_target`
for amplification.

## Sanity checks before reporting

1. `reconstruction_controls.csv` must be shown or discussed; large
   reconstruction-only degradation is a warning about SAE-model
   compatibility.
2. Confirm that every full cell has exactly `--num_latents` selected rows and
   the same number of unique controls in `latent_manifest.csv`.
3. Inspect control match distances rather than dropping poorly matched pairs
   after seeing evaluation effects.
4. Verify that all rows use the intended LoRA checkpoint and new FT-SAE
   checkpoint in `preflight.json` / `latent_manifest.csv`.
5. Keep the latent manifest even if the causal hypothesis is not supported;
   do not reselect features on evaluation outcomes.
