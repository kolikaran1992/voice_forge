from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import pathlib

# Path to the pyx file
pyx_path = "voice_forge/vits_clone/monotonic_align/core.pyx"

extensions = cythonize(
    [
        Extension(
            "voice_forge.vits_clone.monotonic_align.core",
            [pyx_path],
        )
    ],
    compiler_directives={"language_level": "3"},
    annotate=False,
)

setup(
    name="voice-forge",
    version="0.1.0",
    packages=find_packages(include=["voice_forge", "voice_forge.*"]),
    ext_modules=extensions,
    include_package_data=True,
)
