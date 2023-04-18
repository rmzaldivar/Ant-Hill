import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimModule(nn.Module):
    """
    Base class for modules with configurable optimizer hyperparameters.
    """
    # TODO: Implement methods for registering parameters with configurable learning rates and weight decays.

class Sin(nn.Module):
    """
    Sinusoidal function.

    dim: int
        Output tensor dimension.
    w: float
        Frequency of the sine wave.
    """
    # TODO: Implement the Sin class with the specified arguments.

class PositionalEmbedding(nn.Module):
    """
    Positional embedding layer.

    emb_dim: int
        Dimension of the embedding space.
    seq_len: int
        Length of the sequence.
    """
    # TODO: Implement the PositionalEmbedding class with the specified arguments.

class ExponentialModulation(nn.Module):
    """
    Exponential modulation function.

    d_model: int
        Dimension of the model.
    fast_decay_pct: float
        Percentage of the model that decays quickly.
    slow_decay_pct: float
        Percentage of the model that decays slowly.
    """
    # TODO: Implement the ExponentialModulation class with the specified arguments.

class HyenaFilter(nn.Module):
    """
    Hyena filter.

    d_model: int
        Dimension of the model.
    emb_dim: int
        Dimension of the embedding space.
    order: int
        Width of the implicit MLP.
    fused_fft_conv: bool
        Whether to use fused FFT-based convolutions.
    seq_len: int
        Length of the sequence.
    lr: float
        Learning rate.
    lr_pos_emb: float
        Learning rate for the positional embedding layer.
    """
    # TODO: Implement the HyenaFilter class with the specified arguments.
