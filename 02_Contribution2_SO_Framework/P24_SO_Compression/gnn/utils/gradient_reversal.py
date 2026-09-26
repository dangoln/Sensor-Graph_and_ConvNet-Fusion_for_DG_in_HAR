"""
Gradient Reversal Layer (GRL)
==============================
Implements the gradient reversal trick from Ganin et al. (2016):
"Domain-Adversarial Training of Neural Networks"

During forward pass: identity (passes input unchanged).
During backward pass: negates and scales the gradient by -lambda.

This forces the encoder to learn features that are simultaneously:
  - Good for activity classification (positive gradient from task head)
  - Bad for domain identification (reversed gradient from domain head)

Result: domain-invariant feature representations.
"""

import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Autograd function that reverses gradients during backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    """
    Module wrapper for gradient reversal.

    Usage:
        grl = GradientReversalLayer()
        reversed_features = grl(features, lambda_val=0.5)
    """

    def forward(self, x, lambda_val=1.0):
        return GradientReversalFunction.apply(x, lambda_val)


def ganin_lambda_schedule(progress, gamma=10.0):
    """
    Lambda schedule from Ganin et al. (2016).

    Args:
        progress: float in [0, 1], fraction of training completed
        gamma:    sharpness parameter (default 10.0)

    Returns:
        lambda value in [0, 1], following 2/(1+exp(-gamma*p)) - 1
    """
    import math
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


def linear_lambda_schedule(progress, lambda_max=1.0):
    """Simple linear schedule: lambda = lambda_max * progress."""
    return lambda_max * progress
