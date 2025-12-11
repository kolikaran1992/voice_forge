# Packaging & Cython Build Guide

This project contains a Cython extension (`core.pyx`) and must remain
installable both locally and on Google Colab. Below is the minimal working
setup.



## ✅ Local installation (with Cython)

1. **Create `setup.py` at repo root**:

```python
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = cythonize(
    [
        Extension(
            "voice_forge.vits_clone.monotonic_align.core",
            ["voice_forge/vits_clone/monotonic_align/core.pyx"],
        )
    ],
    compiler_directives={"language_level": "3"},
)

setup(
    name="voice-forge",
    version="0.1.0",
    packages=find_packages(include=["voice_forge", "voice_forge.*"]),
    ext_modules=extensions,
    include_package_data=True,
    package_data={"voice_forge": ["settings_file/*.toml"]},
)
```

2. **Create `MANIFEST.in`**

```
recursive-include voice_forge *.pyx *.pxd *.c *.toml
```

3. **Update `pyproject.toml` build backend**

```toml
[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools>=61.0", "wheel", "Cython>=3.0"]
```

4. **Install locally**

```bash
poetry install
poetry run pip install -e .
```



## ✅ Make the package installable on Colab

1. **Commit the generated C file**:

```
voice_forge/vits_clone/monotonic_align/core.c
```

3. Push to GitHub.

4. Install on Colab:

```bash
pip install git+https://github.com/<your-username>/voice_forge.git
```

Colab builds the extension from `core.c` (no Cython needed), and all TOML
settings files are included because of the `MANIFEST.in` + `package_data` fix.



## Result

✔ Local development works ✔ Cython extension builds cleanly ✔ Non-Python files
(TOML) are included in wheels ✔ Colab installation works with different Python
versions





