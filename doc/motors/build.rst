.. _ElectricalBuild:

******************
Build from Source
******************

The electrical motor module (``mujoco.electrical``), the companion native
plugin, and the viewer demos all live in the main MuJoCo repository. There is
no separate checkout for this extension: building MuJoCo from source is enough.

Pull the repository
===================

Clone the repository once:

.. code-block:: shell

   git clone https://github.com/<your-user-or-org>/mujoco.git
   cd mujoco

To update an existing checkout later:

.. code-block:: shell

   git pull --rebase

If you are working from a fork, replace the URL above with your fork URL.

Install system dependencies
===========================

On Ubuntu/Debian, install the build tools and graphics dependencies used by the
MuJoCo build:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install -y \
       git \
       cmake \
       ninja-build \
       python3 \
       python3-pip \
       python3-venv \
       gcc-10 \
       g++-10 \
       libgl1-mesa-dev \
       libwayland-dev \
       libxinerama-dev \
       libxcursor-dev \
       libxkbcommon-dev \
       libxrandr-dev \
       libxi-dev

.. note::

   The current tree requires a modern C++ toolchain. On Ubuntu 20.04, the
   verified working setup is ``gcc-10`` / ``g++-10``.

If you also want the Python demos and notebooks, create a virtual environment:

.. code-block:: shell

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install --upgrade --require-hashes -r python/build_requirements.txt

Clean a previous build
======================

If you want a fully clean rebuild, remove the entire build directory:

.. code-block:: shell

   rm -rf build

If you only want to reset CMake state while keeping the directory, remove the
cache and fetched dependency state:

.. code-block:: shell

   rm -rf build/CMakeCache.txt build/CMakeFiles build/_deps

This is especially helpful after switching compilers or generators (for example,
``Unix Makefiles`` -> ``Ninja``), or when CMake reports a stale
``FetchContent`` / dependency-population error.

Configure and build
===================

Configure the project with the newer compiler toolchain:

.. code-block:: shell

   CC=gcc-10 CXX=g++-10 cmake -S . -B build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF

Build the core library:

.. code-block:: shell

   cmake --build build --parallel --target mujoco

If you also want the desktop viewer executable:

.. code-block:: shell

   cmake --build build --parallel --target simulate

Run the electrical demos
========================

Once the build is complete, the Python electrical demos can be launched from
the repository root:

.. code-block:: shell

   python3 python/examples/electrical/demo.py
   python3 python/examples/electrical/demo_visualizer.py --scenario both

Rebuild the documentation
=========================

To preview the Sphinx documentation locally:

.. code-block:: shell

   cd doc
   python3 -m pip install -r requirements.txt
   make html

The generated HTML is written to ``doc/_build/html/``.
