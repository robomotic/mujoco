# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Shared fixtures for mujoco.electrical tests.

This conftest lives at python/tests/electrical/ — deliberately OUTSIDE the
mujoco package tree so that pytest imports neither python/mujoco/__init__.py
nor the C extension.  A minimal mujoco stub is installed first; the
electrical sub-package is then importable via the python/ root on sys.path,
bypassing the stub's (absent) __init__.py body.
"""

import pathlib
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# 1. Install a minimal 'mujoco' stub BEFORE adding python/ to sys.path.
#    This prevents Python from ever executing python/mujoco/__init__.py,
#    which requires the compiled C extension.
# ---------------------------------------------------------------------------
if "mujoco" not in sys.modules:
    _stub = types.ModuleType("mujoco")
    # __path__ enables sub-package discovery (mujoco.electrical.*).
    _PYTHON_MUJOCO = str(pathlib.Path(__file__).parents[2] / "mujoco")
    _stub.__path__ = [_PYTHON_MUJOCO]
    # xml_integration.py uses TYPE_CHECKING for MjSpec so nothing extra needed.
    sys.modules["mujoco"] = _stub

# ---------------------------------------------------------------------------
# 2. Add python/ to sys.path so 'from mujoco.electrical.X import Y' works.
# ---------------------------------------------------------------------------
_PYTHON_ROOT = str(pathlib.Path(__file__).parents[2])
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

# ---------------------------------------------------------------------------
# Inline motor spec dictionaries — no network access required
# ---------------------------------------------------------------------------

#: Faulhaber 2264W024BP4 — fully specified (all RL + thermal fields populated)
FAULHABER_DICT: dict = {
    "motor_id": "faulhaber_2264w024bp4",
    "manufacturer": "Faulhaber",
    "model": "2264W024BP4",
    "gear_ratio": 1.0,
    "reflected_inertia": 9.2e-7,
    "voltage_range": [0.0, 24.0],
    "resistance": 0.22,
    "inductance": 2.4e-5,
    "motor_constant_kt": 0.0118,
    "motor_constant_ke": 0.0118,
    "stall_torque": 1.311,
    "continuous_torque": 0.059,
    "peak_torque": 1.311,
    "no_load_speed": 2209.6,
    "no_load_current": 0.261,
    "number_of_pole_pairs": 2,
    "max_speed": 1256.6,
    "weight": 0.14,
    "commutation": "Hall",
    "friction_static": 0.00041,
    "friction_dynamic": 1.09e-6,
    "thermal_resistance": 5.0,
    "thermal_time_constant": 950.0,
    "max_winding_temperature": 125.0,
    "rotation_angle_range": [-3.14159, 3.14159],
    "stall_current": 10.0,
    "operating_current": 3.0,
    "ambient_temperature": 25.0,
    "encoder_resolution": 2048,
    "encoder_type": "incremental",
    "protocol": "PWM",
}

#: Unitree A1 — degraded spec (R, L, thermal fields are null)
UNITREE_A1_DICT: dict = {
    "motor_id": "unitree_a1",
    "manufacturer": "Unitree",
    "model": "A1",
    "gear_ratio": 9.1,
    "voltage_range": [0.0, 24.0],
    "resistance": None,
    "inductance": None,
    "motor_constant_kt": 0.0476,
    "motor_constant_ke": 0.0476,
    "stall_torque": 33.5,
    "continuous_torque": 20.0,
    "peak_torque": 33.5,
    "no_load_speed": 52.0,
    "no_load_current": 0.5,
    "thermal_resistance": None,
    "thermal_time_constant": None,
    "max_winding_temperature": None,
    "ambient_temperature": 25.0,
}


@pytest.fixture()
def faulhaber_spec():
    """Fully-specified Faulhaber motor (exercises the full RL + thermal path)."""
    from mujoco.electrical.motor_spec import MotorSpecification
    return MotorSpecification.from_dict(FAULHABER_DICT)


@pytest.fixture()
def unitree_a1_spec():
    """Degraded Unitree A1 motor (exercises the null-R/L path)."""
    from mujoco.electrical.motor_spec import MotorSpecification
    return MotorSpecification.from_dict(UNITREE_A1_DICT)
