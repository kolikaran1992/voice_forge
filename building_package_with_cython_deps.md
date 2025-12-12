# 📦 **Packaging & Cython Build Guide (Final & Corrected)**

This project contains a Cython extension (`core.pyx`) inside:

```
voice_forge/vits_clone/monotonic_align/core.pyx
```

The project must:

* Build locally (with Cython installed)
* Build on Colab/Kaggle (without requiring Cython)
* Install from GitHub using pip
* Include TOML config files inside wheels

Below is the **minimal correct setup**.



# ✅ 1. `setup.py` (minimal, correct, safe)

⚠️ **IMPORTANT:** This file must not contain metadata like `name=`, `version=`,
`packages=` → These are taken from `pyproject.toml` (PEP 621). If you include
them in `setup.py`, it overrides Poetry metadata and breaks extras, wheels, etc.

### ✅ **Final `setup.py`:**

```python
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = cythonize(
    [
        Extension(
            "voice_forge.vits_clone.monotonic_align.core",
            ["voice_forge/vits_clone/monotonic_align/core.pyx"],
        )
    ],
    compiler_directives={"language_level": "3"},
)

setup(ext_modules=ext_modules)
```

**This file does exactly one job:** build the Cython extension.



# ✅ 2. `MANIFEST.in` (include all needed files)

### Final version:

```
recursive-include voice_forge *.pyx *.pxd *.c
recursive-include voice_forge *.toml
```

This ensures:

* Cython sources (`.pyx`, `.pxd`)
* Generated C files (`.c`)
* Your settings files (`settings/*.toml`)

are included in wheels and sdists.



# ✅ 3. `pyproject.toml` (correct build backend for Cython)

You **must** use Poetry-core as the build backend, NOT setuptools.

This ensures:

* PEP 621 metadata support
* Editable installs with pip
* Compatibility with Kaggle
* Proper dependency resolution

### Final version:

```toml
[build-system]
requires = ["poetry-core>=1.6.0", "setuptools>=61.0", "wheel", "Cython>=3.0"]
build-backend = "poetry.core.masonry.api"
```

### Why this works:

* `poetry-core` handles PEP 621 metadata
* `setuptools` gets pulled in only to run `setup.py` for the extension
* Cython is available when needed during editable builds
* Wheels built using `core.c` do not require Cython at install time



# 🧩 What about dependencies?

Your environment-specific dependencies (Colab/Kaggle) belong in your **separate
branches**, not in this packaging guide.

The packaging guide deals only with building the Cython extension correctly.



# 🚀 4. Local installation (editable mode)

```bash
poetry install
poetry run pip install -e .
```

This:

* Installs dependencies via Poetry
* Runs Cython compile via setup.py
* Makes your package editable for development



# 🚀 5. Colab / Kaggle installation (no Cython needed)

Since you committed the generated **`core.c`** file, Cython is **not required**
at install time.

On Colab/Kaggle:

```bash
pip install git+https://github.com/<your-username>/voice_forge.git@kaggle
```

Or:

```bash
pip install git+https://github.com/<your-username>/voice_forge.git@main
```

What happens:

* pip reads `pyproject.toml`
* setuptools builds the C-extension from `core.c`
* TOML settings files are included automatically
* No Cython needed at install time


---