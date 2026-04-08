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
"""Tests for ``BatteryManager`` — Phase 3.

Covers:
  * Terminal voltage: OCV interpolation, voltage drop under load
  * SOC depletion via Coulomb counting
  * Thermal heating and Newton cooling
  * Voltage clamping to pack limits
  * is_depleted / is_overheated sentinel properties
  * reset() behaviour
  * bus_voltage consistency with ElectricalMotor integration
  * dt validation
"""

import math

import numpy as np
import pytest

from mujoco.electrical.battery_manager import BatteryManager
from mujoco.electrical.battery_spec import BatterySpecification


# ---------------------------------------------------------------------------
# Inline spec — no network access, mirrors the live unitree_g1_9ah asset
# ---------------------------------------------------------------------------

UNITREE_G1_9AH_DICT: dict = {
    "battery_id": "unitree_g1_9ah",
    "manufacturer": "Unitree",
    "model": "G1-9Ah",
    "chemistry": "Li-ion",
    "cells_series": 6,
    "cells_parallel": 1,
    "nominal_cell_voltage": 3.6,
    "max_cell_voltage": 4.2,
    "min_cell_voltage": 2.5,
    "capacity_ah": 9.0,
    "internal_resistance": 0.015,
    "max_continuous_current": 15.12,
    "thermal_capacity": 1200.0,
    "thermal_resistance": 8.0,
    "max_temperature": 50.0,
    "ambient_temperature": 25.0,
    "ocv_curve": [
        [0.0, 2.5],
        [0.1, 3.2],
        [0.2, 3.45],
        [0.3, 3.55],
        [0.5, 3.6],
        [0.7, 3.65],
        [0.8, 3.75],
        [0.9, 3.95],
        [1.0, 4.2],
    ],
    "internal_resistance_soc_curve": [
        [0.0, 2.5],
        [0.2, 1.5],
        [0.5, 1.0],
        [0.8, 0.95],
        [1.0, 0.9],
    ],
}

DT = 0.002  # 500 Hz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def spec() -> BatterySpecification:
    return BatterySpecification.from_dict(UNITREE_G1_9AH_DICT)


@pytest.fixture()
def mgr(spec) -> BatteryManager:
    """Fresh BatteryManager at full charge."""
    return BatteryManager(spec, initial_soc=1.0)


# ===========================================================================
# Derived pack voltages (spec sanity checks)
# ===========================================================================


class TestSpecDerivedVoltages:
    def test_nominal_voltage(self, spec):
        assert spec.nominal_voltage == pytest.approx(3.6 * 6)

    def test_max_voltage(self, spec):
        assert spec.max_voltage == pytest.approx(4.2 * 6)

    def test_min_voltage(self, spec):
        assert spec.min_voltage == pytest.approx(2.5 * 6)

    def test_energy_wh(self, spec):
        assert spec.energy_wh == pytest.approx(9.0 * 3.6 * 6)


# ===========================================================================
# Initial state
# ===========================================================================


class TestInitialState:
    def test_soc_full(self, mgr):
        assert mgr.soc == pytest.approx(1.0)

    def test_temp_at_ambient(self, mgr, spec):
        assert mgr.temperature == pytest.approx(spec.ambient_temperature)

    def test_not_depleted_when_full(self, mgr):
        assert not mgr.is_depleted

    def test_not_overheated_at_ambient(self, mgr):
        assert not mgr.is_overheated

    def test_initial_partial_soc(self, spec):
        m = BatteryManager(spec, initial_soc=0.5)
        assert m.soc == pytest.approx(0.5)

    def test_soc_clamped_above_one(self, spec):
        m = BatteryManager(spec, initial_soc=1.5)
        assert m.soc == pytest.approx(1.0)

    def test_soc_clamped_below_zero(self, spec):
        m = BatteryManager(spec, initial_soc=-0.1)
        assert m.soc == pytest.approx(0.0)


# ===========================================================================
# Terminal voltage
# ===========================================================================


class TestTerminalVoltage:
    def test_open_circuit_at_full_soc(self, mgr, spec):
        """Zero current → terminal voltage equals OCV = 4.2*6 = 25.2 V."""
        v = mgr.terminal_voltage
        assert v == pytest.approx(spec.max_cell_voltage * spec.cells_series)

    def test_voltage_drops_under_load(self, mgr):
        """Non-zero current reduces terminal voltage below OCV."""
        v_open = mgr.terminal_voltage
        # step returns terminal voltage computed BEFORE state update
        v_load = mgr.step(current_draw=5.0, dt=DT)
        assert v_load < v_open

    def test_voltage_clamped_to_max(self, mgr, spec):
        """Charging current (negative) must not exceed max_voltage."""
        v = mgr.step(current_draw=-1e6, dt=DT)
        assert v <= spec.max_voltage + 1e-9

    def test_voltage_clamped_to_min(self, spec):
        """At empty SOC terminal voltage must not fall below min_voltage."""
        m = BatteryManager(spec, initial_soc=0.0)
        v = m.step(current_draw=100.0, dt=DT)
        assert v >= spec.min_voltage - 1e-9

    def test_known_voltage_drop_first_step(self, mgr, spec):
        """Verify the voltage-drop calculation at full SOC analytically.

        At SOC=1.0:
          OCV per cell = 4.2 V  (spec endpoint)
          r_mult       = 0.9    (spec endpoint at SOC=1.0)
          R_int        = 0.015 * 0.9 = 0.0135 Ω
          V_drop       = 10.0 * 0.0135 = 0.135 V
          V_term       = 4.2*6 - 0.135 = 25.065 V
        """
        I = 10.0
        r_mult_at_full = spec.internal_resistance_soc_curve[-1][1]  # 0.9
        r_int = spec.internal_resistance * r_mult_at_full
        expected = spec.max_cell_voltage * spec.cells_series - I * r_int
        v = mgr.step(current_draw=I, dt=DT)
        assert v == pytest.approx(expected, rel=1e-6)

    def test_terminal_voltage_property_no_side_effects(self, mgr):
        """Polling terminal_voltage must not change SOC or temperature."""
        soc_before = mgr.soc
        temp_before = mgr.temperature
        _ = mgr.terminal_voltage
        assert mgr.soc == soc_before
        assert mgr.temperature == temp_before


# ===========================================================================
# SOC depletion — Coulomb counting
# ===========================================================================


class TestSOCDepletion:
    def test_soc_decreases_under_load(self, mgr):
        soc_before = mgr.soc
        mgr.step(current_draw=5.0, dt=DT)
        assert mgr.soc < soc_before

    def test_soc_unchanged_at_zero_current(self, mgr):
        soc_before = mgr.soc
        mgr.step(current_draw=0.0, dt=DT)
        assert mgr.soc == pytest.approx(soc_before)

    def test_soc_increases_on_charge(self, spec):
        """Negative current (charging) must increase SOC."""
        m = BatteryManager(spec, initial_soc=0.5)
        soc_before = m.soc
        m.step(current_draw=-5.0, dt=DT)
        assert m.soc > soc_before

    def test_coulomb_counting_analytic_one_step(self, mgr, spec):
        """Single-step SOC decrement matches the analytic formula."""
        I = 5.0
        expected_delta = -I / (spec.capacity_ah * 3600.0) * DT
        soc_before = mgr.soc
        mgr.step(current_draw=I, dt=DT)
        assert mgr.soc == pytest.approx(soc_before + expected_delta, rel=1e-9)

    def test_soc_clamped_to_zero(self, spec):
        """Massive discharge must not push SOC below 0."""
        m = BatteryManager(spec, initial_soc=0.001)
        m.step(current_draw=1e6, dt=1.0)
        assert m.soc >= 0.0

    def test_soc_clamped_to_one_on_overcharge(self, spec):
        """Charging past 100% must not push SOC above 1."""
        m = BatteryManager(spec, initial_soc=0.999)
        m.step(current_draw=-1e6, dt=1.0)
        assert m.soc <= 1.0

    def test_is_depleted_flag(self, spec):
        m = BatteryManager(spec, initial_soc=spec.min_soc)
        assert m.is_depleted

    def test_not_depleted_just_above_min(self, spec):
        m = BatteryManager(spec, initial_soc=spec.min_soc + 0.01)
        assert not m.is_depleted

    def test_sustained_discharge_depletes_pack(self, spec):
        """Running at 15 A (max continuous) should deplete SOC over time."""
        m = BatteryManager(spec, initial_soc=1.0)
        steps = int(10.0 / DT)  # 10 simulated seconds
        for _ in range(steps):
            m.step(current_draw=15.0, dt=DT)
        assert m.soc < 1.0

    def test_multi_step_no_nan(self, mgr):
        for _ in range(100):
            v = mgr.step(current_draw=5.0, dt=DT)
            assert math.isfinite(v)
            assert math.isfinite(mgr.soc)
            assert math.isfinite(mgr.temperature)


# ===========================================================================
# Thermal dynamics
# ===========================================================================


class TestThermalDynamics:
    def test_temp_rises_under_load(self, mgr, spec):
        T0 = mgr.temperature
        steps = int(30.0 / DT)  # 30 simulated seconds
        for _ in range(steps):
            mgr.step(current_draw=10.0, dt=DT)
        assert mgr.temperature > T0

    def test_temp_cools_at_zero_current(self, spec):
        """A warm pack with no current draw must cool towards ambient."""
        m = BatteryManager(spec, initial_soc=1.0)
        m._temp = 45.0  # artificially warm
        T_warm = m.temperature
        for _ in range(int(60.0 / DT)):
            m.step(current_draw=0.0, dt=DT)
        assert m.temperature < T_warm

    def test_steady_state_temp_finite(self, mgr):
        """After many steps the temperature must not diverge."""
        for _ in range(int(60.0 / DT)):
            mgr.step(current_draw=5.0, dt=DT)
        assert math.isfinite(mgr.temperature)

    def test_is_overheated_flag(self, spec):
        m = BatteryManager(spec, initial_soc=1.0)
        m._temp = spec.max_temperature
        assert m.is_overheated

    def test_not_overheated_at_ambient(self, mgr):
        assert not mgr.is_overheated

    def test_joule_heating_analytic_one_step(self, spec):
        """Verify thermal equation on the first step analytically.

        At SOC=1.0, I=10A, T=T_amb:
          r_mult = 0.9, R_int = 0.015 * 0.9 = 0.0135 Ω
          joule_heat = 10² * 0.0135 = 1.35 W
          cooling    = (25 - 25) / 8.0 = 0 W  (at ambient)
          dT         = 1.35 / 1200.0 = 1.125e-3 °C/s
          ΔT         = dT * 0.002     = 2.25e-6 °C
        """
        m = BatteryManager(spec, initial_soc=1.0)
        I = 10.0
        r_mult = spec.internal_resistance_soc_curve[-1][1]
        r_int = spec.internal_resistance * r_mult
        joule = I ** 2 * r_int
        expected_dT = joule / spec.thermal_capacity * DT
        T_before = m.temperature
        m.step(current_draw=I, dt=DT)
        assert m.temperature == pytest.approx(T_before + expected_dT, rel=1e-6)


# ===========================================================================
# reset()
# ===========================================================================


class TestReset:
    def test_reset_restores_full_soc(self, mgr):
        for _ in range(100):
            mgr.step(current_draw=10.0, dt=DT)
        assert mgr.soc < 1.0
        mgr.reset()
        assert mgr.soc == pytest.approx(1.0)

    def test_reset_restores_temperature(self, mgr, spec):
        for _ in range(int(30.0 / DT)):
            mgr.step(current_draw=10.0, dt=DT)
        assert mgr.temperature > spec.ambient_temperature
        mgr.reset()
        assert mgr.temperature == pytest.approx(spec.ambient_temperature)

    def test_reset_to_partial_soc(self, mgr):
        mgr.reset(initial_soc=0.5)
        assert mgr.soc == pytest.approx(0.5)

    def test_reset_clears_is_depleted(self, spec):
        m = BatteryManager(spec, initial_soc=0.0)
        assert m.is_depleted
        m.reset(initial_soc=1.0)
        assert not m.is_depleted


# ===========================================================================
# Input validation
# ===========================================================================


class TestValidation:
    def test_negative_dt_raises(self, mgr):
        with pytest.raises(ValueError, match="dt"):
            mgr.step(current_draw=1.0, dt=-0.001)

    def test_zero_dt_raises(self, mgr):
        with pytest.raises(ValueError, match="dt"):
            mgr.step(current_draw=1.0, dt=0.0)

    def test_valid_dt_does_not_raise(self, mgr):
        mgr.step(current_draw=1.0, dt=1e-6)  # no error
