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
"""Tests for ``ElectricalMotor`` — Phase 2.

Covers:
  * Full RL-circuit code path (Faulhaber 2264W024BP4)
  * Degraded code path (Unitree A1, null R/L/thermal)
  * Thermal winding dynamics
  * Torque and current clipping
  * bus_voltage override
  * reset() behaviour
  * dt validation
"""

import math

import numpy as np
import pytest

from mujoco.electrical.electrical_motor import ElectricalMotor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DT = 0.002  # 500 Hz — matches the plan's default timestep


# ===========================================================================
# Full RL path — Faulhaber 2264W024BP4
# ===========================================================================


class TestFullRLPath:
    """ElectricalMotor with a fully-specified spec exercises the RL circuit."""

    def test_has_rl_circuit(self, faulhaber_spec):
        """Sanity: the Faulhaber spec enables the RL path."""
        assert faulhaber_spec.has_rl_circuit is True

    def test_has_thermal(self, faulhaber_spec):
        """Sanity: the Faulhaber spec enables thermal dynamics."""
        assert faulhaber_spec.has_thermal is True

    def test_returns_two_floats(self, faulhaber_spec):
        motor = ElectricalMotor(faulhaber_spec)
        result = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.1, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        assert isinstance(result, tuple) and len(result) == 2
        torque, current = result
        assert isinstance(torque, float)
        assert isinstance(current, float)

    def test_zero_error_zero_velocity_near_zero_torque(self, faulhaber_spec):
        """At target with no velocity the effort is 0 → torque ≈ 0."""
        motor = ElectricalMotor(faulhaber_spec)
        torque, current = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.0, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        assert abs(torque) < 1e-6
        assert abs(current) < 1e-6

    def test_torque_clamped_to_peak(self, faulhaber_spec):
        """Large position error must not exceed peak_torque."""
        motor = ElectricalMotor(faulhaber_spec)
        torque, _ = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=1e6, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        assert abs(torque) <= faulhaber_spec.peak_torque + 1e-9

    def test_negative_torque_clamped(self, faulhaber_spec):
        """Large negative position error → torque clamped to −peak_torque."""
        motor = ElectricalMotor(faulhaber_spec)
        torque, _ = motor.compute_control(
            pos=1e6, vel=0.0, pos_tgt=0.0, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        assert torque >= -faulhaber_spec.peak_torque - 1e-9

    def test_back_emf_reduces_current(self, faulhaber_spec):
        """Positive velocity opposes the applied voltage → lower output current."""
        motor_still = ElectricalMotor(faulhaber_spec)
        motor_moving = ElectricalMotor(faulhaber_spec)
        _, I_still = motor_still.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.5, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        _, I_moving = motor_moving.compute_control(
            pos=0.0, vel=100.0, pos_tgt=0.5, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        # Moving motor has back-EMF opposing terminal voltage → less current
        assert I_moving < I_still

    def test_bus_voltage_override_limits_current(self, faulhaber_spec):
        """Reducing bus voltage should reduce achievable torque / current."""
        motor_full = ElectricalMotor(faulhaber_spec)
        motor_low = ElectricalMotor(faulhaber_spec)
        _, I_full = motor_full.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.5, vel_tgt=0.0, effort_tgt=0.0, dt=DT,
            bus_voltage=faulhaber_spec.voltage_range[1],
        )
        _, I_low = motor_low.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.5, vel_tgt=0.0, effort_tgt=0.0, dt=DT,
            bus_voltage=6.0,  # quarter of nominal 24 V
        )
        assert abs(I_low) <= abs(I_full) + 1e-9

    def test_i_prev_is_stored(self, faulhaber_spec):
        """After one step, previous_current should reflect the computed current."""
        motor = ElectricalMotor(faulhaber_spec)
        assert motor.previous_current == 0.0
        _, current = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.5, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        assert motor.previous_current == pytest.approx(current, rel=1e-9)

    def test_reset_clears_state(self, faulhaber_spec):
        """After reset() I_prev and temperature return to initial values."""
        motor = ElectricalMotor(faulhaber_spec)
        for _ in range(10):
            motor.compute_control(
                pos=0.0, vel=0.0, pos_tgt=0.5, vel_tgt=0.0, effort_tgt=0.0, dt=DT
            )
        assert motor.previous_current != 0.0
        motor.reset()
        assert motor.previous_current == 0.0
        assert motor.winding_temperature == pytest.approx(
            faulhaber_spec.ambient_temperature
        )

    def test_effort_feedforward(self, faulhaber_spec):
        """Feed-forward torque at zero PD error should produce non-zero output."""
        motor = ElectricalMotor(faulhaber_spec)
        torque, current = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.0, vel_tgt=0.0,
            effort_tgt=0.01, dt=DT,
        )
        assert torque > 0.0

    def test_thermal_heats_up_under_load(self, faulhaber_spec):
        """Sustained current should increase winding temperature."""
        motor = ElectricalMotor(faulhaber_spec)
        T0 = motor.winding_temperature
        for _ in range(500):
            motor.compute_control(
                pos=0.0, vel=0.0, pos_tgt=0.5, vel_tgt=0.0, effort_tgt=0.0, dt=DT
            )
        assert motor.winding_temperature > T0

    def test_thermal_cools_to_ambient_at_zero_current(self, faulhaber_spec):
        """At zero current the winding must cool towards ambient over time."""
        motor = ElectricalMotor(faulhaber_spec)
        motor._temp = 80.0
        T_before = motor.winding_temperature
        for _ in range(2000):
            motor.compute_control(
                pos=0.0, vel=0.0, pos_tgt=0.0, vel_tgt=0.0, effort_tgt=0.0, dt=DT
            )
        assert motor.winding_temperature < T_before

    def test_multi_step_state_consistency(self, faulhaber_spec):
        """Running 50 steps should not produce NaN/Inf."""
        motor = ElectricalMotor(faulhaber_spec)
        for i in range(50):
            torque, current = motor.compute_control(
                pos=float(i) * 0.01,
                vel=0.05,
                pos_tgt=0.5,
                vel_tgt=0.0,
                effort_tgt=0.0,
                dt=DT,
            )
            assert math.isfinite(torque)
            assert math.isfinite(current)
            assert math.isfinite(motor.winding_temperature)

    def test_rl_circuit_physics_first_step(self, faulhaber_spec):
        """Verify RL circuit equations on a known first step (I_prev=0).

        With pos_tgt=0.1, pos=0, vel=0, dt=0.002, kp=100, kd=10:
          effort_des = 100 * 0.1 = 10.0 N·m
          I_target   = 10.0 / Kt = 10.0 / 0.0118 ≈ 847.46 A  (unclamped)
          back_emf   = Ke * 0 = 0
          V_terminal = I_target*R + L*(I_target)/dt  (I_prev=0)
                     = 847.46*0.22 + 2.4e-5*847.46/0.002
                     ≈ 186.44 + 10.17  ≈ 196.6 V  → clamped to 24 V
          I_actual   = (24 - 0 + 0) / (0.22 + 2.4e-5/0.002)
                     = 24 / (0.22 + 0.012) = 24 / 0.232 ≈ 103.45 A
          torque     = clip(Kt * I_actual, ±peak) = clip(1.221, ±1.311) → 1.221 N·m
        """
        spec = faulhaber_spec
        motor = ElectricalMotor(spec, kp=100.0, kd=10.0)
        torque, current = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.1, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        R = spec.resistance
        L = spec.inductance
        expected_I = 24.0 / (R + L / DT)
        expected_torque = float(np.clip(spec.motor_constant_kt * expected_I,
                                        -spec.peak_torque, spec.peak_torque))
        assert torque == pytest.approx(expected_torque, rel=1e-6)
        assert current == pytest.approx(expected_I, rel=1e-6)


# ===========================================================================
# Degraded path — Unitree A1 (null R / L / thermal)
# ===========================================================================


class TestDegradedPath:
    """ElectricalMotor with a degraded spec (no R/L circuit, no thermal)."""

    def test_has_no_rl_circuit(self, unitree_a1_spec):
        assert unitree_a1_spec.has_rl_circuit is False

    def test_has_no_thermal(self, unitree_a1_spec):
        assert unitree_a1_spec.has_thermal is False

    def test_returns_two_floats(self, unitree_a1_spec):
        motor = ElectricalMotor(unitree_a1_spec)
        result = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.1, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_zero_error_zero_torque(self, unitree_a1_spec):
        motor = ElectricalMotor(unitree_a1_spec)
        torque, current = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.0, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        assert abs(torque) < 1e-9
        assert abs(current) < 1e-9

    def test_torque_clamped_to_peak(self, unitree_a1_spec):
        motor = ElectricalMotor(unitree_a1_spec)
        torque, _ = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=1e6, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        assert abs(torque) <= unitree_a1_spec.peak_torque + 1e-9

    def test_current_equals_torque_over_kt(self, unitree_a1_spec):
        """Degraded path: current = torque / Kt (exact relationship)."""
        motor = ElectricalMotor(unitree_a1_spec)
        torque, current = motor.compute_control(
            pos=0.0, vel=0.0, pos_tgt=0.2, vel_tgt=0.0, effort_tgt=0.0, dt=DT
        )
        expected_current = torque / unitree_a1_spec.motor_constant_kt
        assert current == pytest.approx(expected_current, rel=1e-9)

    def test_i_prev_stays_zero(self, unitree_a1_spec):
        """Degraded path does not accumulate I_prev state."""
        motor = ElectricalMotor(unitree_a1_spec)
        for _ in range(10):
            motor.compute_control(
                pos=0.0, vel=0.0, pos_tgt=0.3, vel_tgt=0.0, effort_tgt=0.0, dt=DT
            )
        assert motor.previous_current == 0.0

    def test_temperature_stays_at_ambient(self, unitree_a1_spec):
        """No thermal model → temperature is frozen at ambient."""
        motor = ElectricalMotor(unitree_a1_spec)
        T0 = motor.winding_temperature
        for _ in range(100):
            motor.compute_control(
                pos=0.0, vel=0.0, pos_tgt=0.5, vel_tgt=0.0, effort_tgt=0.0, dt=DT
            )
        assert motor.winding_temperature == pytest.approx(T0)

    def test_reset_no_op_on_frozen_temp(self, unitree_a1_spec):
        """reset() returns temperature to ambient (which it already is)."""
        motor = ElectricalMotor(unitree_a1_spec)
        motor.reset()
        assert motor.winding_temperature == pytest.approx(
            unitree_a1_spec.ambient_temperature
        )

    def test_multi_step_no_nan(self, unitree_a1_spec):
        motor = ElectricalMotor(unitree_a1_spec)
        for i in range(50):
            torque, current = motor.compute_control(
                pos=float(i) * 0.01,
                vel=0.1,
                pos_tgt=0.5,
                vel_tgt=0.0,
                effort_tgt=0.0,
                dt=DT,
            )
            assert math.isfinite(torque)
            assert math.isfinite(current)


# ===========================================================================
# Input validation
# ===========================================================================


class TestValidation:
    def test_negative_dt_raises(self, faulhaber_spec):
        motor = ElectricalMotor(faulhaber_spec)
        with pytest.raises(ValueError, match="dt"):
            motor.compute_control(0.0, 0.0, 0.0, 0.0, 0.0, dt=-0.001)

    def test_zero_dt_raises(self, faulhaber_spec):
        motor = ElectricalMotor(faulhaber_spec)
        with pytest.raises(ValueError, match="dt"):
            motor.compute_control(0.0, 0.0, 0.0, 0.0, 0.0, dt=0.0)

    def test_valid_dt_does_not_raise(self, faulhaber_spec):
        motor = ElectricalMotor(faulhaber_spec)
        motor.compute_control(0.0, 0.0, 0.0, 0.0, 0.0, dt=1e-6)  # no error
