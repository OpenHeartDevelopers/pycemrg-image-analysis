# pycemrg-image-analysis

A high-level framework for cardiac image segmentation post-processing and model creation, built on SimpleITK. Maintained by the [Cardiac Electromechanics Research Group (CEMRG)](https://www.cemrg.com/) at Imperial College London.

---

## Installation

This library depends on [`pycemrg`](https://pypi.org/project/pycemrg/), which provides the core label management and configuration scaffolding utilities. Both packages should be installed together.

### 1. Clone both repositories

```bash
git clone https://github.com/OpenHeartDevelopers/pycemrg.git
git clone https://github.com/OpenHeartDevelopers/pycemrg-image-analysis.git
```

### 2. Create and activate a virtual environment

**Preferred:** Conda 
```bash
conda create -n pycemrg-image-analysis python=3.10 -y 
conda activate pycemrg-image-analysis
```

Or via python's venv 
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

### 3. Install both packages in editable mode

Install `pycemrg` first, then `pycemrg-image-analysis`:

```bash
pip install -e pycemrg/
pip install -e pycemrg-image-analysis/
```

> **Why editable (`-e`)?** Editable installs link directly to the source tree so changes to either package take effect immediately without reinstalling.

---

## Running the tests

```bash
# Unit tests (no data required)
pytest pycemrg-image-analysis/tests/unit/

# Integration tests (requires segmentation data)
PYCEMRG_TEST_DATA_ROOT=/path/to/your/data pytest pycemrg-image-analysis/tests/integration/

# All tests
pytest pycemrg-image-analysis/
```

Integration tests are skipped automatically when `PYCEMRG_TEST_DATA_ROOT` is not set.

---

## Requirements

- Python ≥ 3.10
- SimpleITK ≥ 2.4.0
- `pycemrg` ≥ 0.1.0 (see above)

All other dependencies are declared in `pyproject.toml` and installed automatically by `pip`.
