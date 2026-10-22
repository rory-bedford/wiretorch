# Build the documentation: `make html` once (warnings are errors, as on Read the Docs),
# `make preview` to serve it and rebuild on every change.
SPHINX = uv run --only-group docs

.PHONY: html preview clean

html:
	$(SPHINX) sphinx-build -W . _build/html

preview:
	$(SPHINX) sphinx-autobuild . _build/html

clean:
	rm -rf _build autoapi
