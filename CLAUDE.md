# eoio — Claude Code guidance

## Project overview

*eoio* is an Earth Observation IO library that provides readers for EO data products
and processors for transforming them. The public API is `eoio.read()` in `eoio/interface.py`.

## Required reading before writing readers or processors

When adding or modifying a reader, follow the design documented in:
`docs/content/developer_guide/readers/readers.rst`

When adding or modifying a processor, follow the design documented in:
`docs/content/developer_guide/processors/processors.rst`

All output datasets must conform to the controlled vocabulary in:
`docs/content/developer_guide/controlled_vocabulary.rst`

These are not guidelines — they are requirements. Do not deviate from the patterns
described there without a clear reason.

## Key conventions

### Readers

- Readers are **thin orchestrators**. Heavy IO, geometry, and parsing logic lives in
  helper modules (`layout.py`, `data_io.py`, `aux_data.py`, `conventions.py`, `metadata/`).
- Inherit from `BaseRasterReader` (gridded imagery), `BaseInSituReader` (point/time-series),
  or `BaseReader` (anything else). All are in `eoio/readers/base.py`.
- Public entrypoint is `reader.open()` — implement `open_dataset()`.
- Override `default_vars_sel`, `meas_def`, `aux_def`, `mask_def` to declare available variables.
- Use `eoio.deps` lazy helpers (`lazy_rasterio()`, `lazy_rioxarray()`, etc.) for optional
  imports — never import them at module level.
- Register new readers in `eoio/readers/factory.py` with a regex pattern and a lazy import.

### Processors

- Processors subclass `processor_tools.BaseProcessor` (not `eoio.processors.base.BaseProcessor`,
  which is legacy and unused).
- Register with `@register_processor("name")` from `eoio/processors/registry.py`.
- Constructor signature: `__init__(self, params, context)`. Parse and validate `params` in
  `__init__`; store as instance attributes. Call `super().__init__(context=context)`.
- Implement `run(self, ds: xr.Dataset) -> xr.Dataset`.
- Add an import of the processor module in `eoio/processors/__init__.py` so it is registered
  on package import.

### Output datasets

- Must declare `ds.attrs["Conventions"] = "CF-1.8"`.
- Platform, instrument, and processing_level attributes must use the controlled tokens
  defined in the controlled vocabulary doc.
- Dimension names follow the `x_<resolution>` / `y_<resolution>` pattern for multi-resolution
  raster datasets (e.g. `x_10m`, `y_300m`).
- Mask variables must use `obsarray` flag convention.

## Testing

- Place tests in a `tests/` sub-package within the reader or processor directory.
- Only tests that require real EO product files should be skipped in CI — use
  appropriate pytest markers for those.
- Run `ruff check eoio` and `pytest` before committing.

## Publication pipeline

Never commit with Claud as a co-author

### Remotes

| Remote   | URL | Purpose |
|----------|-----|---------|
| `origin` | `gitlab.npl.co.uk:eco/tools/eomatch` | Primary repo — full commit history, active development |
| `github` | `github.com:meteor-toolkit/eomatch` | Public mirror — orphan `main` with no history |

### Release process

1. Merge the feature branch into GitLab `main` and push:
   ```bash
   git checkout main && git merge <branch>
   git push origin main
   ```
2. Cherry-pick only the new commits onto GitLab `release` (do **not** merge — that pulls in the full history):
   ```bash
   git checkout release
   git cherry-pick <commit(s)>
   git push origin release
   ```
3. Push GitLab `release` to GitHub as `main`:
   ```bash
   git push meteor release:main
   ```
4. Move the `v*` tag to the latest release commit and force-push:
   ```bash
   git tag -f v<X.Y> <commit>
   git push meteor v<X.Y> --force
   ```

### PyPI publishing

Triggered by the `publish.yml` GitHub Actions workflow, either on a `v*` tag push or manually via `workflow_dispatch` (type `publish` in the confirmation input).

The workflow: runs tests (must pass) → builds sdist + wheel → publishes to PyPI via OIDC trusted publishing (no API token required).

**One-time PyPI setup:** the trusted publisher must be registered at https://pypi.org/manage/account/publishing/ before the first upload with:

| Field | Value |
|-------|-------|
| PyPI project name | `eomatch` |
| Owner | `meteor-toolkit` |
| Repository | `eomatch` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

A force-pushed tag does **not** re-trigger the workflow — re-run it manually from the GitHub Actions UI.
