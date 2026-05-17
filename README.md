# eoio

Earth Observation IO: readers for EO data products.

`eoio` provides a common user-friendly interface for reading satellite and
in-situ data from different data sources, with data subsetting flexibility
and basic pre-processing options.

> **Warning:** This software is in beta. Results should be used with
> caution. Please share any feedback via the issue tracker.

## Usage

### Installation

Install the package and its core dependencies:

```bash
pip install -e .
```

Optional extras are available depending on your use case:

```bash
pip install -e ".[dev]"   # Development tools (ruff, mypy, pytest, …)
pip install -e ".[docs]"  # Documentation build (sphinx, …)
```

### Development

Install the pre-commit hooks after cloning:

```bash
pre-commit install
```

When you commit, `ruff` will lint and format your code. If it makes
corrections the commit will be aborted so you can review the changes — just
commit again once you are happy.

Run the test suite with:

```bash
pytest
```

## Compatibility

`eoio` requires Python 3.11 or later and is tested on Python 3.11, 3.12,
and 3.13.

## Licence

`eoio` is released under the GNU Lesser General Public License v3 (LGPLv3).
fSee the [LICENSE](https://github.com/meteor-toolkit/eoio/blob/main/LICENSE) file for the full licence text.

## Authors

`eoio` is developed and maintained by the
[MetEOR Toolkit Team](mailto:team@comet-toolkit.org).
