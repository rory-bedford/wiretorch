# Development

## Setting up

wiretorch is developed with [uv](https://docs.astral.sh/uv/). PyTorch comes from an
extra, `cu129` or `cpu`; pass the same one to every `uv run`, or uv swaps in PyPI's
torch.

```bash
git clone https://github.com/rory-bedford/wiretorch && cd wiretorch
uv sync --extra cu129 --group dev        # or --extra cpu
uv run --no-sync pre-commit install      # ruff on commit, the test suite on push
```

## Tests and benchmarks

```bash
uv run --extra cu129 pytest                          # GPU tests skip without CUDA
uv run --extra cu129 python benchmarks/speed.py      # also: current, rate
```

`tests/README.md` explains how the tests work and when to regenerate their fixtures.

## Documentation

From `docs/`, `make preview` serves the site at <http://127.0.0.1:8000> and rebuilds it
on every change; `make html` builds it once, with warnings as errors. Tutorials are
committed with their outputs; after changing one, re-execute them with
`uv run --extra cu129 python docs/run_tutorials.py`.

## CI

Every push and pull request runs, on GitHub:

- ruff (lint and format),
- the tests on CPU (GPU tests skip), with coverage sent to
  [Codecov](https://codecov.io/gh/rory-bedford/wiretorch),
- the documentation build.

Read the Docs rebuilds the `latest` docs on every push to `main`.

## Releasing

A version tag on the public repo publishes that version. For `0.1.0b2`:

1. Set the version, `uv version --no-sync 0.1.0b2`, and add a `## [0.1.0b2] - <date>` section
   to `CHANGELOG.md`. Commit.
2. Publish it to the public repo, then tag that commit and push the tag:

   ```bash
   git tag v0.1.0b2
   git push public v0.1.0b2
   ```

The tag runs the tests, publishes the package to PyPI and makes a GitHub release with
the changelog section as its notes; the release stops if the tag doesn't match the
version in `pyproject.toml`. Read the Docs builds the docs for the tag.

Versions follow [PEP 440](https://peps.python.org/pep-0440/): betas are `0.1.0b1`,
`0.1.0b2`, and so on, and pip installs them by default until `0.1.0` is out.

## Conventions

Commit messages are one line starting with a [gitmoji](https://gitmoji.dev/), for
example `:bug: Fix the reset of tracked variables`.

## Colours

| | |
|---|---|
| Background | `#263640` (slate) |
| Primary | `#5096c0` (blue) |
| Accents | `#2a9d8f` (teal), `#e9c46a` (yellow) |
| Highlight | `#e87878` (red) |

Excitatory neurons are red and inhibitory ones blue. {py:mod}`wiretorch.visualization.theme`
has the colours and a light and dark Matplotlib theme (`viz.use_theme()`).
`docs/make_icons.py` makes the logo and favicons from `assets/wiretorch-logo.png`.
