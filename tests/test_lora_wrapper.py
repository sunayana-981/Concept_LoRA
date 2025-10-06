import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Insert project root (one level up from tests/) to sys.path so imports like "CLIP_LoRA" work.
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from CLIP_LoRA.loralib.wrappers import LoRAWrapper  # Assuming the wrapper is defined here

class TestLoRAWrapper:
    def setup_method(self):
        # Initialize the frozen matrix B1 and trainable matrices A1, A2, B2
        self.B1 = torch.randn(10, 10)  # Example frozen matrix
        self.A1 = nn.Parameter(torch.randn(10, 10))  # Trainable matrix A1
        self.A2 = nn.Parameter(torch.randn(10, 10))  # Trainable matrix A2
        self.B2 = nn.Parameter(torch.randn(10, 10))  # Trainable matrix B2
        self.wrapper = LoRAWrapper(self.A1, self.A2, self.B1, self.B2)

    def test_lora_matrix_computation(self):
        # Compute the LoRA matrix
        lora_matrix = self.wrapper.compute_lora_matrix()
        
        # Expected computation: A1B1 + A2B2
        expected_matrix = self.A1 @ self.B1 + self.A2 @ self.B2
        
        # Check if the computed matrix is close to the expected matrix
        assert torch.allclose(lora_matrix, expected_matrix, atol=1e-6)

    def test_trainable_parameters(self):
        # Check if A1, A2, and B2 are trainable
        assert self.A1.requires_grad
        assert self.A2.requires_grad
        assert self.B2.requires_grad
        
        # Check if B1 is not trainable
        assert not self.B1.requires_grad

    def test_shape_of_lora_matrix(self):
        # Check the shape of the computed LoRA matrix
        lora_matrix = self.wrapper.compute_lora_matrix()
        assert lora_matrix.shape == (10, 10)  # Expected shape

# To run the tests, use pytest in the command line:
# pytest tests/test_lora_wrapper.py