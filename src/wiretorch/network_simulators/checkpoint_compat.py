"""Loading checkpoints written by earlier versions of the networks."""

#: Buffers older checkpoints contain that current networks no longer have.
#: (``tau_ref``: networks now take it from the cell parameters, unsaved.)
LEGACY_BUFFERS = ("tau_ref",)


def drop_legacy_buffers(module, state_dict, prefix, *args) -> None:
    """``load_state_dict`` pre-hook: discard ``LEGACY_BUFFERS`` so strict loads pass."""
    for name in LEGACY_BUFFERS:
        state_dict.pop(prefix + name, None)
