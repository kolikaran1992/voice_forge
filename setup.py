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
