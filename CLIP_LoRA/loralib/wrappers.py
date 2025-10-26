import torch
import torch.nn as nn

class LoRAWrapper(nn.Module):
    """
    Wrapper that computes LoRA matrix as A1 @ B1 + A2 @ B2.

    - B1 is registered as a buffer (frozen).
    - A1, A2, B2 are registered as trainable nn.Parameter.
    """
    def __init__(self, A1, A2, B1, B2):
        super().__init__()

        # ensure A1, A2, B2 are nn.Parameter (trainable)
        if isinstance(A1, nn.Parameter):
            self.A1 = A1
        else:
            self.A1 = nn.Parameter(A1.clone().detach())

        if isinstance(A2, nn.Parameter):
            self.A2 = A2
        else:
            self.A2 = nn.Parameter(A2.clone().detach())

        if isinstance(B2, nn.Parameter):
            self.B2 = B2
        else:
            self.B2 = nn.Parameter(B2.clone().detach())

        # B1 is frozen: register as buffer (no grad)
        if isinstance(B1, nn.Parameter):
            b1 = B1.detach().clone()
        else:
            b1 = B1.detach().clone()
        self.register_buffer("B1", b1)

    def compute_lora_matrix(self):
        """
        Returns A1 @ B1 + A2 @ B2
        """
        return self.A1 @ self.B1 + self.A2 @ self.B2

    def forward(self, *args, **kwargs):
        # forward returns the composed LoRA matrix by default
        return self.compute_lora_matrix()