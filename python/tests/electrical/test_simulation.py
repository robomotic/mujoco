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
"""Tests for ``SingleEnvSimulation`` and ``SimulationState`` — Phase 4b.

Two tiers:

1. **XML discovery** — ``SingleEnvSimulation._discover_specs`` uses only
   :mod:`xml.etree.ElementTree`, no MuJoCo C extension needed.  All tests
   in :class:`TestDiscoverSpecs` run in every environment.

2. **Full integration** — ``from_xml``, ``step``, ``reset`` require
   ``mujoco.MjModel`` / ``mujoco.mj_step``.  These tests are automatically
   *skipped* when the C extension is not available.
"""

import math

import numpy as np
import pytest

from mujoco.electrical.simulation import SimulationState
from mujoco.electrical.simulation import SingleEnvSimulation


# ---------------------------------------------------------------------------
# Helper: detect whether the real MuJoCo C extension is available.
# The conftest stub installs a bare 'mujoco' module that satisfies an
# importorskip check but lacks MjSpec — so we test for that attribute.
# ---------------------------------------------------------------------------

def _mujoco_c_available() -> bool:
    try:
        import mujoco  # noqa: PLC0415
        return hasattr(mujoco, "MjSpec")
    except ImportError:
        return False


_SKIP_NO_MUJOCO = pytest.mark.skipif(
    not _mujoco_c_available(),
    reason="MuJoCo C extension (MjSpec) not available",
)


# ---------------------------------------------------------------------------
# Inline XML fixtures
# ---------------------------------------------------------------------------

_FAULHABER_XML = """\
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05 0.1"/>
    </body>
  </worldbody>
  <actuator><motor name="a0" joint="j0"/></actuator>
  <custom>
    <text name="motor_a0"     data="motor_spec:faulhaber_2264w024bp4"/>
    <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
  </custom>
</mujoco>
"""

_UNITREE_XML = """\
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05 0.1"/>
    </body>
  </worldbody>
  <actuator><motor name="a0" joint="j0"/></actuator>
  <custom>
    <text name="motor_a0"     data="motor_spec:unitree_a1"/>
    <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
  </custom>
</mujoco>
"""

_MULTI_ACTUATOR_XML = """\
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05 0.1"/>
      <body pos="0 0 0.2">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.04 0.08"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="a0" joint="j0"/>
    <motor name="a1" joint="j1"/>
  </actuator>
  <custom>
    <text name="motor_a0"     data="motor_spec:faulhaber_2264w024bp4"/>
    <text name="motor_a1"     data="motor_spec:faulhaber_2264w024bp4"/>
    <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
  </custom>
</mujoco>
"""

_NO_BATTERY_XML = """\
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05 0.1"/>
    </body>
  </worldbody>
  <actuator><motor name="a0" joint="j0"/></actuator>
  <custom>
    <text name="motor_a0" data="motor_spec:faulhaber_2264w024bp4"/>
  </custom>
</mujoco>
"""

_EMPTY_CUSTOM_XML = """\
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05 0.1"/>
    </body>
  </worldbody>
  <actuator><motor name="a0" joint="j0"/></actuator>
</mujoco>
"""


# ===========================================================================
# Tier 1 — XML auto-discovery (no mujoco C extension required)
# ===========================================================================


class TestDiscoverSpecs:
    """Tests for ``SingleEnvSimulation._discover_specs`` (pure ElementTree)."""

    def test_discovers_single_motor(self):
        motor_map, _ = SingleEnvSimulation._discover_specs(_FAULHABER_XML)
        assert motor_map == {"a0": "faulhaber_2264w024bp4"}

    def test_discovers_single_battery(self):
        _, batt_map = SingleEnvSimulation._discover_specs(_FAULHABER_XML)
        assert batt_map == {"main": "unitree_g1_9ah"}

    def test_discovers_multiple_motors(self):
        motor_map, _ = SingleEnvSimulation._discover_specs(_MULTI_ACTUATOR_XML)
        assert motor_map == {
            "a0": "faulhaber_2264w024bp4",
            "a1": "faulhaber_2264w024bp4",
        }

    def test_no_battery_returns_empty_battery_map(self):
        _, batt_map = SingleEnvSimulation._discover_specs(_NO_BATTERY_XML)
        assert batt_map == {}

    def test_no_custom_section_returns_empty_maps(self):
        motor_map, batt_map = SingleEnvSimulation._discover_specs(
            _EMPTY_CUSTOM_XML
        )
        assert motor_map == {}
        assert batt_map == {}

    def test_ignores_unrelated_text_tags(self):
        xml = """\
<mujoco>
  <custom>
    <text name="foo" data="bar"/>
    <text name="motor_a0" data="motor_spec:faulhaber_2264w024bp4"/>
  </custom>
</mujoco>"""
        motor_map, batt_map = SingleEnvSimulation._discover_specs(xml)
        assert motor_map == {"a0": "faulhaber_2264w024bp4"}
        assert batt_map == {}

    def test_degraded_motor_spec_discovered(self):
        motor_map, _ = SingleEnvSimulation._discover_specs(_UNITREE_XML)
        assert motor_map == {"a0": "unitree_a1"}


# ===========================================================================
# SimulationState dataclass sanity
# ===========================================================================


class TestSimulationState:
    def test_fields_accessible(self):
        s = SimulationState(
            qpos=np.array([0.1]),
            qvel=np.array([0.0]),
            torques=np.array([0.5]),
            currents=np.array([10.0]),
            soc=0.95,
            bus_voltage=24.0,
            temperatures=np.array([26.0]),
            time=0.002,
        )
        assert s.soc == pytest.approx(0.95)
        assert s.bus_voltage == pytest.approx(24.0)
        assert s.time == pytest.approx(0.002)
        assert s.torques[0] == pytest.approx(0.5)

    def test_nan_battery_fields_allowed(self):
        s = SimulationState(
            qpos=np.zeros(1),
            qvel=np.zeros(1),
            torques=np.zeros(1),
            currents=np.zeros(1),
            soc=float("nan"),
            bus_voltage=float("nan"),
            temperatures=np.zeros(1),
        )
        assert math.isnan(s.soc)
        assert math.isnan(s.bus_voltage)


# ===========================================================================
# Tier 2 — Full integration (requires mujoco C extension)
# ===========================================================================


@pytest.fixture()
def faulhaber_sim():
    """SingleEnvSimulation built from Faulhaber XML + live database."""
    if not _mujoco_c_available():
        pytest.skip("MuJoCo C extension not available")
    return SingleEnvSimulation.from_xml(_FAULHABER_XML)


@pytest.fixture()
def unitree_sim():
    """SingleEnvSimulation built from Unitree A1 XML (degraded motor)."""
    if not _mujoco_c_available():
        pytest.skip("MuJoCo C extension not available")
    return SingleEnvSimulation.from_xml(_UNITREE_XML)


@_SKIP_NO_MUJOCO
class TestFromXml:
    def test_n_actuators_single(self, faulhaber_sim):
        assert faulhaber_sim.n_actuators == 1

    def test_n_actuators_multi(self):
        sim = SingleEnvSimulation.from_xml(_MULTI_ACTUATOR_XML)
        assert sim.n_actuators == 2

    def test_battery_loaded(self, faulhaber_sim):
        assert faulhaber_sim.battery is not None
        assert faulhaber_sim.battery.soc == pytest.approx(1.0)

    def test_no_battery_is_none(self):
        sim = SingleEnvSimulation.from_xml(_NO_BATTERY_XML)
        assert sim.battery is None

    def test_motor_is_faulhaber(self, faulhaber_sim):
        assert faulhaber_sim.motors[0].spec.motor_id == "faulhaber_2264w024bp4"

    def test_motor_is_unitree(self, unitree_sim):
        assert unitree_sim.motors[0].spec.motor_id == "unitree_a1"

    def test_model_and_data_exposed(self, faulhaber_sim):
        assert faulhaber_sim.model is not None
        assert faulhaber_sim.data is not None

    def test_unknown_actuator_raises(self):
        xml = """\
<mujoco>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05 0.1"/>
    </body>
  </worldbody>
  <actuator><motor name="a0" joint="j0"/></actuator>
  <custom>
    <text name="motor_nonexistent" data="motor_spec:faulhaber_2264w024bp4"/>
  </custom>
</mujoco>"""
        with pytest.raises(ValueError, match="nonexistent"):
            SingleEnvSimulation.from_xml(xml)


@_SKIP_NO_MUJOCO
class TestStep:
    def test_returns_simulation_state(self, faulhaber_sim):
        state = faulhaber_sim.step(np.array([0.5]))
        assert isinstance(state, SimulationState)

    def test_state_arrays_correct_shape(self, faulhaber_sim):
        state = faulhaber_sim.step(np.array([0.5]))
        assert state.torques.shape == (1,)
        assert state.currents.shape == (1,)
        assert state.temperatures.shape == (1,)

    def test_time_advances(self, faulhaber_sim):
        state = faulhaber_sim.step(np.array([0.5]))
        assert state.time == pytest.approx(0.002)

    def test_torque_clamped_to_peak(self, faulhaber_sim):
        peak = faulhaber_sim.motors[0].spec.peak_torque
        state = faulhaber_sim.step(np.array([1e6]))
        assert abs(state.torques[0]) <= peak + 1e-9

    def test_soc_decreases_under_load(self, faulhaber_sim):
        soc0 = faulhaber_sim.battery.soc
        for _ in range(10):
            faulhaber_sim.step(np.array([0.5]))
        assert faulhaber_sim.battery.soc < soc0

    def test_bus_voltage_finite(self, faulhaber_sim):
        state = faulhaber_sim.step(np.array([0.5]))
        assert math.isfinite(state.bus_voltage)

    def test_no_battery_soc_is_nan(self):
        sim = SingleEnvSimulation.from_xml(_NO_BATTERY_XML)
        state = sim.step(np.array([0.5]))
        assert math.isnan(state.soc)
        assert math.isnan(state.bus_voltage)

    def test_no_nan_in_state_multi_step(self, faulhaber_sim):
        for _ in range(50):
            state = faulhaber_sim.step(np.array([0.1]))
            assert math.isfinite(state.torques[0])
            assert math.isfinite(state.currents[0])
            assert math.isfinite(state.soc)
            assert math.isfinite(state.bus_voltage)
            assert math.isfinite(state.temperatures[0])

    def test_degraded_motor_step(self, unitree_sim):
        """Unitree A1 (degraded) must also step without NaN."""
        state = unitree_sim.step(np.array([0.1]))
        assert math.isfinite(state.torques[0])
        assert math.isfinite(state.currents[0])

    def test_multi_actuator_step(self):
        sim = SingleEnvSimulation.from_xml(_MULTI_ACTUATOR_XML)
        state = sim.step(np.array([0.1, 0.2]))
        assert state.torques.shape == (2,)
        assert state.currents.shape == (2,)

    def test_default_vel_targets_zero(self, faulhaber_sim):
        """Omitting vel_targets is equivalent to passing zeros."""
        import copy
        sim2 = SingleEnvSimulation.from_xml(_FAULHABER_XML)

        s1 = faulhaber_sim.step(np.array([0.5]))
        s2 = sim2.step(np.array([0.5]), vel_targets=np.zeros(1))
        assert s1.torques[0] == pytest.approx(s2.torques[0], rel=1e-6)

    def test_effort_feedforward_changes_torque(self, faulhaber_sim):
        """Non-zero effort_tgt at zero error must shift the output torque."""
        sim2 = SingleEnvSimulation.from_xml(_FAULHABER_XML)
        s_no_ff = faulhaber_sim.step(
            np.array([0.0]), effort_targets=np.array([0.0])
        )
        s_ff = sim2.step(
            np.array([0.0]), effort_targets=np.array([0.001])
        )
        assert s_ff.torques[0] > s_no_ff.torques[0]


@_SKIP_NO_MUJOCO
class TestReset:
    def test_reset_restores_time(self, faulhaber_sim):
        faulhaber_sim.step(np.array([0.5]))
        assert faulhaber_sim.data.time > 0
        faulhaber_sim.reset()
        assert faulhaber_sim.data.time == pytest.approx(0.0)

    def test_reset_restores_battery_soc(self, faulhaber_sim):
        for _ in range(100):
            faulhaber_sim.step(np.array([0.5]))
        assert faulhaber_sim.battery.soc < 1.0
        faulhaber_sim.reset()
        assert faulhaber_sim.battery.soc == pytest.approx(1.0)

    def test_reset_clears_motor_integrator(self, faulhaber_sim):
        for _ in range(10):
            faulhaber_sim.step(np.array([0.5]))
        faulhaber_sim.reset()
        assert faulhaber_sim.motors[0].previous_current == 0.0
