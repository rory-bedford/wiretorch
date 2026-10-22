"""Loading checkpoints written by earlier versions of the networks."""

#: Buffers older checkpoints contain that current networks no longer have.
LEGACY_BUFFERS = ("tau_ref",)  # the refractory period, never simulated


def drop_legacy_buffers(module, state_dict, prefix, *args) -> None:
    """``load_state_dict`` pre-hook: discard ``LEGACY_BUFFERS`` so strict loads pass."""
    for name in LEGACY_BUFFERS:
        state_dict.pop(prefix + name, None)
