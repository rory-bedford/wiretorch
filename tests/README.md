# Tests

```bash
uv run --extra cu129 pytest                    # everything; GPU tests skip without CUDA
uv run --extra cpu pytest -m "not gpu"         # what CI runs (GitHub, every push)
```

The full suite also runs before every `git push` once the hooks are installed
(`uv run --no-sync pre-commit install`): it is the only place the GPU tests run.

The simulators are chaotic, so "the same simulator" needs a careful definition. The
suite compares against **fixtures recorded from the simulators before any optimisation**
(`fixtures/*.npz`, ~3 MB), at four levels of strictness:

| File | Checks | Strictness |
|---|---|---|
| `test_lockstep.py` | Restart from every recorded state and take `WINDOW` steps: the definition of correct. Floating-point differences cannot compound, so spikes must match exactly (bar a neuron within 1 µV of threshold) and the state to a tolerance. Runs on CPU and GPU. | exact spikes |
| `test_free_run.py` | A 200 ms run from rest reproduces the recorded spike trains. | exact, CPU |
| `test_gradients.py` | Surrogate gradients of a fixed random spike readout, through 100 steps of backprop through time. | 1e-4 (CPU) |
| `test_training.py` | Four `SNNTrainer` updates follow the recorded parameter trajectory. | 1e-4 |
| `test_invariants.py` | Chunking, batch independence, `reset_state`, tracking, run-time threshold — the simulator against itself. | exact |
| `test_statistics.py` | Large networks on the GPU, eager, compiled and with float16 storage, for both families: recurrent firing statistics agree with the recording at least as well as two recordings with different inputs agree with each other; feedforward spike trains agree to 0.1%. | statistical |
| `test_fixtures.py` | Every rebuilt network still matches the fingerprint it was recorded with. | exact |

The networks (`networks.py`) are small, seeded, and built through the public API, so
nothing depends on data outside the repo. `harness.py` holds the code that drives them,
shared by the tests and the fixture generator so a recording and its check do exactly
the same thing.

## Regenerating fixtures

`python tests/fixtures/generate.py` (and `... statistics`, on a GPU) rewrites them from
the current code; `... scenarios <name> ...` and `... statistics <fixture>` rewrite only
the ones named. Do this only after a change that is *meant* to alter the dynamics —
never to make a failing test pass. Each fixture records the commit it came from, and
`test_fixtures.py` rejects one recorded from uncommitted library code.

The current-based fixture `current_feedforward` keeps the spike trains, voltages and
gradients recorded from the original current-based simulator; only its lockstep states
(`state.*`) were added later, from the PyTorch reference, which reproduced the original
recordings bit for bit (`current_recurrent` too, before it moved to a regime whose rates
hold across network size). The other current-based fixtures were recorded from the
reference.
