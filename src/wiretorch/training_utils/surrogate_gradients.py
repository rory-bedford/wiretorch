"""Custom heaviside step function with surrogate gradient for spiking neurons.

Uses the setup_context pattern for torch.compile compatibility (PyTorch >= 2.0).
"""

import torch


class SurrGradSpike(torch.autograd.Function):
    """
    Spiking nonlinearity with surrogate gradient.

    Forward: Heaviside step function (1 if input > 0, else 0).
    Backward: Normalized negative part of a fast sigmoid (Zenke & Ganguli, 2018).

    Compatible with torch.compile via the setup_context pattern.
    """

    @staticmethod
    def forward(input, scale):
        return (input > 0).to(input.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, scale = inputs
        ctx.save_for_backward(input)
        ctx.scale = scale

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad = grad_output / (ctx.scale * torch.abs(input) + 1.0) ** 2
        return grad, None


class SampleSpike(torch.autograd.Function):
    """Bernoulli-sampled spikes in forward, surrogate gradient in backward.

    Forward: Compute logistic sigmoid probability, then sample from Bernoulli.
    Backward: Use the derivative of the logistic sigmoid (scale * sigma * (1 - sigma)).

    Compatible with torch.compile via the setup_context pattern.
    """

    @staticmethod
    def forward(input, scale):
        prob = torch.sigmoid(scale * input)
        return torch.bernoulli(prob)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, scale = inputs
        ctx.save_for_backward(input)
        ctx.scale = scale

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        prob = torch.sigmoid(ctx.scale * input)
        grad = grad_output * ctx.scale * prob * (1.0 - prob)
        return grad, None
