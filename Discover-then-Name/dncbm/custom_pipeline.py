from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from jaxtyping import Int64
from pydantic import NonNegativeInt, PositiveInt, validate_call
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from sparse_autoencoder.metrics.abstract_metric import MetricLocation, MetricResult


from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
    ParameterUpdateResults,
)

from time import time

import torch
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossReductionType
from sparse_autoencoder.metrics.metrics_container import MetricsContainer, default_metrics
from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.tensor_types import Axis


if TYPE_CHECKING:
    from sparse_autoencoder.metrics.abstract_metric import MetricResult


class Pipeline:
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    optimizer: AbstractOptimizerWithReset
    """Optimizer to use."""

    progress_bar: tqdm | None
    """Progress bar for the pipeline."""

    total_activations_trained_on: int = 0
    """Total number of activations trained on state."""

    @property
    def n_components(self) -> int:
        """Number of source model components the SAE is trained on."""

        return 1  # since we are training on a single component, which is the out layer

    def __init__(
        self,
        activation_resampler: AbstractActivationResampler | None,
        autoencoder: SparseAutoencoder,
        loss: AbstractLoss,
        optimizer: AbstractOptimizerWithReset,
        checkpoint_directory: Path = None,
        log_frequency: PositiveInt = 100,
        metrics: MetricsContainer = default_metrics,
        device: torch.cuda = 'cuda',
        args=None
    ) -> None:

        self.activation_resampler = activation_resampler
        self.autoencoder = autoencoder
        self.checkpoint_directory = checkpoint_directory
        self.log_frequency = log_frequency
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.device = device
        self.args = args

    @validate_call(config={"arbitrary_types_allowed": True})
    def train_autoencoder(
        self, activation_store: TensorActivationStore, train_batch_size: PositiveInt
    ) -> Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]:
        """Train the sparse autoencoder.

        Args:
            activation_store: Activation store from the generate section.
            train_batch_size: Train batch size.

        Returns:
            Number of times each neuron fired, for each component.
        """

        activations_dataloader = DataLoader(
            activation_store,
            batch_size=train_batch_size,
            shuffle=True
        )

        learned_activations_fired_count: Int64[
            Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.zeros(
            (self.n_components, self.autoencoder.n_learned_features),
            dtype=torch.int64,
            device=self.device,)

        for id, store_batch in enumerate(activations_dataloader):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Move the batch to the device (in place)
            batch = store_batch.detach().to(self.device)

            # Forward pass
            learned_activations, reconstructed_activations = self.autoencoder.forward(
                batch)

            # Get loss & metrics
            metrics: list[MetricResult] = []
            total_loss, loss_metrics = self.loss.scalar_loss_with_log(
                batch,
                learned_activations,
                reconstructed_activations,
                component_reduction=LossReductionType.MEAN
            )
            metrics.extend(loss_metrics)

            with torch.no_grad():
                for metric in self.metrics.train_metrics:
                    calculated = metric.calculate(
                        TrainMetricData(batch, learned_activations,
                                        reconstructed_activations)
                    )
                    metrics.extend(calculated)

            # Store count of how many neurons have fired
            with torch.no_grad():
                fired = learned_activations > 0
                learned_activations_fired_count.add_(fired.sum(dim=0))

            # Backwards pass
            total_loss.backward()
            self.optimizer.step()
            self.autoencoder.post_backwards_hook()

            # Log training metrics
            self.total_activations_trained_on += train_batch_size
            if (
                wandb.run is not None
                and int(self.total_activations_trained_on / train_batch_size) % self.log_frequency
                == 0
            ):
                log = {}
                for metric_result in metrics:
                    log.update(metric_result.wandb_log)
                wandb.log(
                    log,
                    step=self.total_activations_trained_on,
                    commit=False,
                )
        return learned_activations_fired_count

    def save_checkpoint(self, *, is_final: bool = False) -> Path:
        """Save the model as a checkpoint.

        Args:
            is_final: Whether this is the final checkpoint.

        Returns:
            Path to the saved checkpoint.
        """
        # Create the name
        name: str = f"sparse_autoencoder_{'final' if is_final else self.total_activations_trained_on}"
        safe_name = quote_plus(name, safe="_")
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        file_path: Path = self.checkpoint_directory / f"{safe_name}.pt"

        print(f"DEBUG: Attempting to save checkpoint to {file_path} (is_final={is_final})")
        try:
            torch.save(self.autoencoder.state_dict(), file_path)
        except Exception as e:
            print(f"ERROR: Failed to save checkpoint to {file_path}: {e}")
            raise
        else:
            print(f"DEBUG: Checkpoint saved -> {file_path}")
        return file_path

    def update_parameters(self, parameter_updates: list[ParameterUpdateResults]) -> None:
        """Update the parameters of the model from the results of the resampler.

        Args:
            parameter_updates: Parameter updates from the resampler.
        """
        for component_idx, component_parameter_update in enumerate(parameter_updates):
            # Update the weights and biases
            self.autoencoder.encoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_weight_updates,
                component_idx=component_idx,
            )
            self.autoencoder.encoder.update_bias(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_bias_updates,
                component_idx=component_idx,
            )
            self.autoencoder.decoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_decoder_weight_updates,
                component_idx=component_idx,
            )

            # Reset the optimizer
            for parameter, axis in self.autoencoder.reset_optimizer_parameter_details:
                self.optimizer.reset_neurons_state(
                    parameter=parameter,
                    neuron_indices=component_parameter_update.dead_neuron_indices,
                    axis=axis,
                    component_idx=component_idx,
                )

    def get_activation_store(self, activation_fname):
        activations = torch.load(activation_fname)
        activation_store = TensorActivationStore(
            activations.shape[0], self.autoencoder.n_input_features, self.n_components)
        activation_store.empty()
        activation_store.extend(activations, component_idx=0)
        return activation_store

    # considering train_val_fnames to contain a single fname
    def validation(self, activation_store, train_batch_size):
        activations_dataloader = DataLoader(
            activation_store, batch_size=train_batch_size, shuffle=True)

        with torch.no_grad():
            total_losses = torch.zeros((4, len(activations_dataloader)))
            with tqdm(desc="Validation", total=len(activations_dataloader),) as progress_bar:
                for batch_id, store_batch in enumerate(activations_dataloader):
                    batch = store_batch.detach().to(self.device)
                    # Forward pass
                    learned_activations, reconstructed_activations = self.autoencoder.forward(
                        batch)
                    _, loss_metrics = self.loss.scalar_loss_with_log(
                        batch,
                        learned_activations,
                        reconstructed_activations,
                        component_reduction=LossReductionType.MEAN
                    )
                    for loss_id, loss_metric in enumerate(loss_metrics):
                        total_losses[loss_id,
                                     batch_id] = loss_metric.component_wise_values
                    mean_losses = total_losses.mean(dim=1)
                    progress_bar.update(1)
                return loss_metrics, mean_losses

    def run_pipeline(
        self,
        train_batch_size: PositiveInt,
        val_frequency: NonNegativeInt | None = None,
        checkpoint_frequency: NonNegativeInt | None = None,
        num_epochs=None,
        train_fnames=None,
        train_val_fnames=None,
        start_time=0,
        resample_epoch_freq: NonNegativeInt = 0,
    ) -> None:
        """Run the full training pipeline.

        Behaviour:
        - iterate epochs
        - for each train activation shard: load ActivationStore, train on it
        - optionally run resampler (if provided)
        - optionally validate using single train_val file (if provided)
        - optionally save intermediate checkpoints (based on checkpoint_frequency)
        - always save final checkpoint at the end
        """
        if num_epochs is None:
            num_epochs = 1

        train_fnames = train_fnames or []
        train_val_fnames = train_val_fnames or []

        print(f"DEBUG: Starting pipeline: num_epochs={num_epochs}, n_train_files={len(train_fnames)}, n_val_files={len(train_val_fnames)}")

        for epoch in range(num_epochs):
            epoch_start = time()
            print(f"DEBUG: Epoch {epoch} start at {epoch_start - start_time} seconds")

            # Iterate over each shard of activations
            for shard_idx, train_fname in enumerate(train_fnames):
                shard_load_start = time()
                print(f"DEBUG: Loading activation shard [{shard_idx}] {train_fname} at {shard_load_start - start_time} seconds")
                activation_store = None
                try:
                    activation_store = self.get_activation_store(train_fname)
                except Exception as e:
                    print(f"ERROR: Failed to create activation store from {train_fname}: {e}")
                    continue

                # try to determine number of samples for debug
                try:
                    n_samples = len(activation_store)
                except Exception:
                    n_samples = getattr(activation_store, "n_activations", "unknown")
                print(f"DEBUG: {train_fname}: {n_samples} NUM SAMPLES.")

                # Train on this activation shard
                shard_train_start = time()
                fired_counts = None
                try:
                    fired_counts = self.train_autoencoder(activation_store, train_batch_size)
                except Exception as e:
                    print(f"ERROR: Training failed on shard {train_fname}: {e}")
                shard_train_end = time()
                print(f"DEBUG: Training completed for shard [{shard_idx}] at {shard_train_end - start_time} seconds")

                # Optionally run the activation resampler (defensive)
                if self.activation_resampler is not None:
                    try:
                        if hasattr(self.activation_resampler, "maybe_resample"):
                            print(f"DEBUG: Calling activation_resampler.maybe_resample() at {time()-start_time} seconds")
                            self.activation_resampler.maybe_resample()
                        elif hasattr(self.activation_resampler, "resample"):
                            print(f"DEBUG: Calling activation_resampler.resample() at {time()-start_time} seconds")
                            self.activation_resampler.resample()
                        else:
                            print(f"DEBUG: activation_resampler present but no known resample API; skipping at {time()-start_time} seconds")
                    except Exception as e:
                        print(f"ERROR: activation_resampler call failed: {e}")
                else:
                    print(f"DEBUG: No activation_resampler configured; resampling skipped at {time()-start_time} seconds")

                # Free activation store to release memory
                try:
                    del activation_store
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"DEBUG: Activation store deleted at {time()-start_time} seconds")
                except Exception as e:
                    print(f"WARNING: Failed to delete activation store cleanly: {e}")

                # Validation: if a train_val file exists, run validation after this shard if requested
                if val_frequency is not None and val_frequency != 0 and train_val_fnames:
                    try:
                        # current code expects a single train_val file; use first
                        val_fname = train_val_fnames[0]
                        print(f"DEBUG: Running validation on {val_fname} at {time()-start_time} seconds")
                        val_store = self.get_activation_store(val_fname)
                        val_metrics, val_mean_losses = self.validation(val_store, train_batch_size)
                        print(f"DEBUG: Validation completed at {time()-start_time} seconds; mean_losses={val_mean_losses}")
                        del val_store
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"ERROR: Validation failed: {e}")
                else:
                    print(f"DEBUG: Validation skipped at {time()-start_time} seconds")

                # Checkpointing based on total activations seen
                if checkpoint_frequency is not None and checkpoint_frequency > 0:
                    try:
                        if (self.total_activations_trained_on > 0 and
                                self.total_activations_trained_on % checkpoint_frequency == 0):
                            ckpt = self.save_checkpoint(is_final=False)
                            print(f"DEBUG: Intermediate checkpoint saved to {ckpt} at {time()-start_time} seconds")
                        else:
                            print(f"DEBUG: Checkpoint save skipped at {time()-start_time} seconds")
                    except Exception as e:
                        print(f"ERROR: Checkpoint saving failed: {e}")
                else:
                    print(f"DEBUG: Checkpoint frequency unset/zero; checkpoint save skipped at {time()-start_time} seconds")

            epoch_end = time()
            print(f"DEBUG: Epoch {epoch} completed at {epoch_end - start_time} seconds")

            # Optionally perform resample at epoch granularity (if requested)
            if resample_epoch_freq and (epoch + 1) % max(1, resample_epoch_freq) == 0:
                try:
                    if self.activation_resampler is not None and hasattr(self.activation_resampler, "resample_epoch"):
                        print(f"DEBUG: Performing epoch-level resample at {time()-start_time} seconds")
                        self.activation_resampler.resample_epoch()
                    else:
                        print(f"DEBUG: No epoch-level resample API available; skipping at {time()-start_time} seconds")
                except Exception as e:
                    print(f"ERROR: Epoch-level resample failed: {e}")

        # Save final checkpoint
        try:
            saved = self.save_checkpoint(is_final=True)
            print(f"DEBUG: Final checkpoint written to: {saved}")
        except Exception as e:
            print(f"ERROR: Final checkpoint failed: {e}")
