# MuJoCo Python Bindings

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/mujoco
[pypi-badge]: https://badge.fury.io/py/mujoco.svg
[pypi]: https://pypi.org/project/mujoco/

This package is the canonical Python bindings for the
[MuJoCo physics engine](https://github.com/google-deepmind/mujoco).
These bindings are developed and maintained by Google DeepMind, and is kept
up-to-date with the latest developments in MuJoCo itself.

The `mujoco` package provides direct access to raw MuJoCo C API functions,
structs, constants, and enumerations. Structs are provided as Python classes,
with Pythonic initialization and deletion semantics.

It is not the aim of this package to provide fully fledged
scene/environment/game authoring API, as there are already a number of existing
packages that do this well. However, this package does provide a number of
lower-level components outside of MuJoCo itself that are likely to be useful to
most users who access MuJoCo through Python. For example, the `egl`, `glfw`, and
`osmesa` subpackages contain utilities for setting up OpenGL rendering contexts.

## Installation

The recommended way to install this package is via [PyPI](https://pypi.org/project/mujoco/):

```sh
pip install mujoco
```

A copy of the MuJoCo library is provided as part of the package and does **not**
need to be downloaded or installed separately.

### Source

**IMPORTANT:** Building from source is only necessary if you are modifying the
Python bindings (or are trying to run on exceptionally old Linux systems).
If that's not the case, then we recommend installing the prebuilt binaries from
PyPI.

If you need to build the Python bindings from source, please consult
[the documentation](https://mujoco.readthedocs.io/en/latest/python.html#building-from-source).

If you want to build a wheel from this repository, from the repo root run:

```sh
export TMPDIR=/tmp
python3.11 -m venv /tmp/mujoco-wheel-venv
source /tmp/mujoco-wheel-venv/bin/activate
python -m pip install --upgrade --require-hashes -r python/build_requirements.txt
python -m pip install --upgrade --require-hashes -r python/build_requirements_usd.txt

cmake -S . -B build-wheel -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DCMAKE_INSTALL_PREFIX="$TMPDIR/mujoco_install" \
  -DMUJOCO_BUILD_EXAMPLES=OFF
cmake --build build-wheel --config Release --parallel
cmake --install build-wheel

mkdir -p "$TMPDIR/mujoco_install/mujoco_plugin"
cp build-wheel/lib/libactuator* "$TMPDIR/mujoco_install/mujoco_plugin/"
cp build-wheel/lib/libelasticity* "$TMPDIR/mujoco_install/mujoco_plugin/"
cp build-wheel/lib/libobj_decoder* "$TMPDIR/mujoco_install/mujoco_plugin/"
cp build-wheel/lib/libstl_decoder* "$TMPDIR/mujoco_install/mujoco_plugin/"
cp build-wheel/lib/libsensor* "$TMPDIR/mujoco_install/mujoco_plugin/"
cp build-wheel/lib/libsdf_plugin* "$TMPDIR/mujoco_install/mujoco_plugin/"

(cd python && bash ../.github/workflows/build_steps.sh make_python_sdist)
(cd python/dist && \
  MUJOCO_PATH="$TMPDIR/mujoco_install" \
  MUJOCO_PLUGIN_PATH="$TMPDIR/mujoco_install/mujoco_plugin" \
  MUJOCO_CMAKE_ARGS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF -G Ninja" \
  pip wheel -v --no-deps mujoco-*.tar.gz)
```

The resulting wheel will be written to `python/dist/`. Use Python `>=3.10`
(`3.11` recommended) and GCC 10+ or Clang 13+.

## Usage

Once installed, the package can be imported via `import mujoco`. Please consult
our [documentation](https://mujoco.readthedocs.io/en/stable/python.html) for
further detail on the package's API.

We recommend going through the tutorial notebook which covers the basics of
MuJoCo using Python:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)

## Versioning

The `major.minor.micro` portion of the version number matches the version of
MuJoCo that the bindings provide. Optionally, if we release updates to the
Python bindings themselves that target the same version of MuJoCo, a `.postN`
suffix is added, for example `2.1.2.post2` represents the second update to the
bindings for MuJoCo 2.1.2.

## License and Disclaimer

Copyright 2022 DeepMind Technologies Limited

MuJoCo and its Python bindings are licensed under the Apache License,
Version 2.0. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
