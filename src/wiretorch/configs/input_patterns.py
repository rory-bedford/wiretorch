"""Input pattern configuration: how each feedforward cell type's rates are patterned.

One config per feedforward cell type. Each assembly gets its own input pattern: the
inputs most strongly connected to the assembly fire above baseline, the rest below it,
so that the population mean stays at baseline (see
``dataloaders.input_patterns.generate_assembly_firing_rates``). An Ornstein-Uhlenbeck
process then mixes the patterns over time (``dataloaders.rate_processes``).
"""

from .base import Config


class InputPatternConfig(Config):
    """Baseline and modulated firing rates of one feedforward cell type.

    The most strongly connected ``modulation_fraction`` of inputs is modulated up by
    ``modulation_rate``; the rest are modulated down, keeping the mean at baseline
    (see ``dataloaders.input_patterns.modulated_rates``).

    If ``baseline_variance`` > 0, each pattern's mean firing rate is sampled from
    N(baseline_rate, baseline_variance) instead of being fixed at baseline_rate. This
    relaxes the perfect-whitening constraint and introduces per-pattern variation in
    total feedforward drive.
    """

    baseline_rate: float
    modulation_rate: float
    modulation_fraction: float
    baseline_variance: float = 0.0

    def get_modulated_rates(self, n_neurons: int) -> tuple[float, float]:
        """``(up_rate, down_rate)`` in Hz for ``n_neurons`` inputs, as
        ``generate_assembly_firing_rates`` assigns them (the mean stays at baseline)."""
        from wiretorch.dataloaders.input_patterns import modulated_rates

        _, up, down = modulated_rates(
            self.baseline_rate,
            self.modulation_rate,
            self.modulation_fraction,
            n_neurons,
        )
        return up, down

    def get_n_modulated(self, n_neurons: int) -> int:
        """Number of the ``n_neurons`` inputs modulated up (the rest go down)."""
        return int(n_neurons * self.modulation_fraction)

    def to_dict(self) -> dict:
        """The four rate parameters as a plain dict."""
        return {
            "baseline_rate": self.baseline_rate,
            "modulation_rate": self.modulation_rate,
            "modulation_fraction": self.modulation_fraction,
            "baseline_variance": self.baseline_variance,
        }


#: Parameter tables that may hold the input patterns, newest name first ("odours" is
#: the name older experiments use).
INPUT_PATTERN_TABLES = ("input_patterns", "odours")
#: Keys of that table that parametrise the OU mixing process, not a cell type.
RATE_PROCESS_KEYS = ("tau", "temperature", "sigma")


def load_input_patterns(params: dict) -> tuple[dict, dict[str, InputPatternConfig]]:
    """Read an experiment's input-pattern table.

    Returns ``(rate_process, configs)``: the OU process's 'tau', 'temperature' and
    'sigma', and one :class:`InputPatternConfig` per feedforward cell type.
    """
    table = next(params[name] for name in INPUT_PATTERN_TABLES if name in params)
    rate_process = {key: table[key] for key in RATE_PROCESS_KEYS}
    configs = {
        name: InputPatternConfig(**config)
        for name, config in table.items()
        if name not in RATE_PROCESS_KEYS
    }
    return rate_process, configs
