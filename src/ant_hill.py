import torch.nn as nn
from .modules import PositionalEmbedding, HyenaFilter, ExponentialModulation

class AntHill(nn.Module):
    """
    Ant Hill model that integrates the Hyena Hierarchy and Spatial Skew-Gaussian Process.
    """

    def __init__(self, d_model, emb_dim, order, fused_fft_conv, seq_len, lr, lr_pos_emb):
        super().__init__()

        # Initialize layers
        self.positional_embedding = PositionalEmbedding(emb_dim, seq_len)
        self.hyena_filter = HyenaFilter(d_model, emb_dim, order, fused_fft_conv, seq_len, lr, lr_pos_emb)
        self.exponential_modulation = ExponentialModulation(d_model, fast_decay_pct=0.5, slow_decay_pct=0.5)

    def forward(self, x, mask=None):
        """
        Forward pass for Ant Hill model.

        x: torch.Tensor
            Input tensor.
        mask: torch.Tensor, optional
            Mask tensor.

        Returns:
            torch.Tensor: Model output.
        """
        # Apply positional embedding
        x = x + self.positional_embedding(x)

        # Pass input through the Hyena filter
        x = self.hyena_filter(x, mask=mask)

        # Apply exponential modulation
        x = self.exponential_modulation(x)

        return x
