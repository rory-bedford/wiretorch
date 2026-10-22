# Code and checkpoints from connectome_snns

wiretorch was called `connectome_snns`. That name still works:

```python
import connectome_snns.network_simulators.rate as rate    # a DeprecationWarning, once
rate.RecurrentRateNetwork is wt.RecurrentRateNetwork      # True: the same module
```

Every `connectome_snns.X` is the module `wiretorch.X`, so old imports work, and whole
models pickled under the old name load as wiretorch classes. Older module paths such as
`connectome_snns.network_simulators.conductance_based.simulator` still import, and
checkpoints written by the old network classes load into the current ones.

`CONNECTOME_SNNS_BACKEND` is still read if `WIRETORCH_BACKEND` is unset, and experiment
metadata written before the rename (`connectome_snns_commit`) is still understood.

To migrate, replace `connectome_snns` with `wiretorch` in imports, and the dependency
`connectome-snns` with `wiretorch` in `pyproject.toml`.
