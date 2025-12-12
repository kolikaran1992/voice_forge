from setuptools import setup, Extension
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

setup(ext_modules=extensions)
