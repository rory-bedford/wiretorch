# TODO

## Test and tune the kernels on other GPUs

The Triton kernels have only been run on a Quadro RTX 5000 (Turing, compute
capability 7.5). For each newer generation (Ampere, e.g. A100 or RTX 30; Ada, e.g. L40
or RTX 40; Hopper, e.g. H100; Blackwell, e.g. B200 or RTX 50):

- Run the test suite (`uv run pytest`) and check the kernels match the PyTorch
  reference.
- Run `benchmarks/speed.py` and add the results to `docs/performance.md`.
- Tune the block sizes, which were chosen on Turing, e.g. with `triton.autotune` or a
  per-architecture table, and check that state held in registers doesn't spill.
- Check float16 storage on each, including tensor-core use in rate networks.

Also try AMD GPUs (ROCm), including the sparse spike input path.
