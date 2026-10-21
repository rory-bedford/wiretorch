"""Two-layer SNN: recurrent hidden layer + feedforward visible layer.

Wraps a ``ConductanceLIFNetwork`` (Layer 1, hidden neurons) and a
``FeedforwardConductanceLIFNetwork`` (Layer 2, visible neurons) into a
single ``nn.Module`` that can be used with ``SNNTrainer``.

Layer 1 receives ``[FF_input, teacher_visible_spikes]`` and produces
hidden spikes with hidden→hidden recurrence (detached gradients).
Layer 2 receives ``[FF_input, hidden_spikes, teacher_visible_spikes]``
and produces visible spikes. Gradients flow from the loss through Layer
2 into the hidden spikes from Layer 1, giving FF→hidden weights a
gradient signal.

Parameter sharing across layers is automatic when the same Projection
objects are passed to both layers' constructors (Projections own their
parameters and are referenced, not copied). No special wrapper-side
sharing logic is needed.
"""

import torch
import torch.nn as nn


class TwoLayerSNN(nn.Module):
    """Two-layer SNN: recurrent hidden layer + feedforward visible layer.

    Args:
        layer1: ``ConductanceLIFNetwork`` for hidden neurons. Receives the
            full ``input_spikes`` tensor (FF + visible teacher spikes).
        layer2: ``FeedforwardConductanceLIFNetwork`` for visible neurons.
            Receives ``[FF_input, hidden_spikes, visible_teacher_spikes]``.
        n_ff: Number of true feedforward inputs (first n_ff columns of
            ``input_spikes``). Used to split the input tensor for Layer 2.
        return_hidden_spikes: If True, ``forward`` always returns
            ``{"spikes": visible_spikes, "hidden_spikes": hidden_spikes}``
            regardless of ``track_variables``.
    """

    def __init__(
        self,
        layer1,
        layer2,
        n_ff,
        return_hidden_spikes=False,
    ):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.n_ff = n_ff
        self.return_hidden_spikes = return_hidden_spikes
        # Of Layer 2's inputs, only the hidden spikes carry a gradient.
        layer2.differentiable_inputs = slice(n_ff, n_ff + layer1.n_neurons)

    def forward(self, input_spikes):
        """Run both layers sequentially.

        Args:
            input_spikes: ``(batch, time, n_ff + n_visible)`` from collate.

        Returns:
            When ``track_variables=False``:
                ``Tensor`` of visible spikes ``(batch, time, n_visible)``.
            When ``track_variables=True``:
                ``dict`` with ``"spikes"`` and ``"hidden_spikes"`` plus any
                tracked Layer 1 variables under ``hidden_*`` keys.
        """
        # Layer 1: hidden neurons (recurrent)
        layer1_out = self.layer1(input_spikes)
        if isinstance(layer1_out, dict):
            hidden_spikes = layer1_out["spikes"]
        else:
            hidden_spikes = layer1_out

        # Build Layer 2 input: [FF, hidden, visible_teacher]
        layer2_input = torch.cat(
            [
                input_spikes[:, :, : self.n_ff],
                hidden_spikes,
                input_spikes[:, :, self.n_ff :],
            ],
            dim=2,
        )

        # Layer 2: visible neurons (feedforward)
        layer2_out = self.layer2(layer2_input)

        if self.track_variables:
            if isinstance(layer2_out, dict):
                result = layer2_out
            else:
                result = {"spikes": layer2_out}
            result["hidden_spikes"] = hidden_spikes
            if isinstance(layer1_out, dict):
                for key, value in layer1_out.items():
                    if key != "spikes":
                        result[f"hidden_{key}"] = value
            return result
        elif self.return_hidden_spikes:
            return {"spikes": layer2_out, "hidden_spikes": hidden_spikes}
        else:
            return layer2_out

    def reset_state(self, batch_size=None):
        self.layer1.reset_state(batch_size)
        self.layer2.reset_state(batch_size)

    def get_checkpoint_state(self):
        l1 = self.layer1.get_checkpoint_state()
        l2 = self.layer2.get_checkpoint_state()
        return {f"layer1_{k}": v for k, v in l1.items()} | {
            f"layer2_{k}": v for k, v in l2.items()
        }

    def load_checkpoint_state(self, state):
        l1 = {
            k.removeprefix("layer1_"): v
            for k, v in state.items()
            if k.startswith("layer1_")
        }
        l2 = {
            k.removeprefix("layer2_"): v
            for k, v in state.items()
            if k.startswith("layer2_")
        }
        self.layer1.load_checkpoint_state(l1)
        self.layer2.load_checkpoint_state(l2)

    @property
    def track_variables(self):
        return self.layer2.track_variables

    @track_variables.setter
    def track_variables(self, value):
        self.layer1.track_variables = value
        self.layer2.track_variables = value

    @property
    def track_batch_idx(self):
        return self.layer2.track_batch_idx

    @track_batch_idx.setter
    def track_batch_idx(self, value):
        self.layer1.track_batch_idx = value
        self.layer2.track_batch_idx = value

    @property
    def device(self):
        return self.layer1.device

    @property
    def scaling_factors(self):
        """Recurrent scaling factors (from Layer 1 hidden→hidden)."""
        return self.layer1.scaling_factors

    @property
    def scaling_factors_FF(self):
        """Feedforward scaling factors (from Layer 1 FF→hidden)."""
        return self.layer1.scaling_factors_FF

    @property
    def surrgrad_scale(self):
        return self.layer1.surrgrad_scale

    @surrgrad_scale.setter
    def surrgrad_scale(self, value):
        self.layer1.surrgrad_scale = value
        self.layer2.surrgrad_scale = value

    @property
    def batch_size(self):
        return self.layer1.batch_size
