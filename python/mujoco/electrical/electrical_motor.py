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
"""Single-environment electrical motor simulation.

Two code paths are selected at runtime based on the spec:

* **Full RL path** (``spec.has_rl_circuit is True``): integrates the
  RL-circuit dynamics, computes back-EMF, clips the terminal voltage to the
  supply rail, and derives the *actual* current.  Used for fully-specified
  motors such as ``faulhaber_2264w024bp4``.

* **Degraded path** (``spec.has_rl_circuit is False``): skips RL dynamics
  and directly clamps the desired effort.  The ``unitree_a1`` spec (null
  R/L) exercises this path.

Thermal winding dynamics are computed independently when
``spec.has_thermal is True``.
"""

from __future__ import annotations

import numpy as np

from mujoco.electrical.motor_spec import MotorSpecification


class ElectricalMotor:
  """NumPy single-environment electrical motor.

  Args:
    spec: Motor specification loaded from a JSON asset.
    kp:   Position-error proportional gain (N·m/rad).
    kd:   Velocity-error derivative gain (N·m·s/rad).
  """

  def __init__(
      self,
      spec: MotorSpecification,
      kp: float = 100.0,
      kd: float = 10.0,
  ) -> None:
    self.spec = spec
    self.kp = kp
    self.kd = kd
    # Mutable integrator state — reset on episode boundaries.
    self._I_prev: float = 0.0
    self._temp: float = float(spec.ambient_temperature)

  # ------------------------------------------------------------------ #
  # Public API                                                           #
  # ------------------------------------------------------------------ #

  def reset(self) -> None:
    """Reset integrator state for a new episode."""
    self._I_prev = 0.0
    self._temp = float(self.spec.ambient_temperature)

  def compute_control(
      self,
      pos: float,
      vel: float,
      pos_tgt: float,
      vel_tgt: float,
      effort_tgt: float,
      dt: float,
      bus_voltage: float | None = None,
  ) -> tuple[float, float]:
    """Compute the output torque and winding current for one time step.

    Args:
      pos:         Current joint position (rad).
      vel:         Current joint velocity (rad/s).
      pos_tgt:     Desired joint position (rad).
      vel_tgt:     Desired joint velocity (rad/s).
      effort_tgt:  Feed-forward torque (N·m).
      dt:          Simulation time step (s).  Must be > 0.
      bus_voltage: Overrides ``spec.voltage_range[1]`` when given (V).

    Returns:
      ``(torque, current)`` where *torque* is the output torque applied to
      the joint (N·m) and *current* is the estimated winding current (A).

    Raises:
      ValueError: If *dt* is not positive.
    """
    if dt <= 0.0:
      raise ValueError(f"dt must be positive, got {dt!r}")

    s = self.spec
    Kt = s.motor_constant_kt
    Ke = s.motor_constant_ke
    v_min = s.voltage_range[0]
    v_max = bus_voltage if bus_voltage is not None else s.voltage_range[1]

    effort_des = self.kp * (pos_tgt - pos) + self.kd * (vel_tgt - vel) + effort_tgt

    if s.has_rl_circuit:
      torque, current = self._full_rl_step(
          vel=vel,
          effort_des=effort_des,
          Kt=Kt,
          Ke=Ke,
          R=s.resistance,   # type: ignore[arg-type]
          L=s.inductance,   # type: ignore[arg-type]
          v_min=v_min,
          v_max=v_max,
          peak_torque=s.peak_torque,
          dt=dt,
      )
    else:
      torque, current = self._degraded_step(
          effort_des=effort_des,
          Kt=Kt,
          peak_torque=s.peak_torque,
      )

    if s.has_thermal and s.resistance is not None:
      self._update_thermal(current=current, dt=dt)

    self._I_prev = current if s.has_rl_circuit else 0.0
    return torque, current

  # ------------------------------------------------------------------ #
  # Read-only state properties                                           #
  # ------------------------------------------------------------------ #

  @property
  def winding_temperature(self) -> float:
    """Current winding temperature (°C)."""
    return self._temp

  @property
  def previous_current(self) -> float:
    """Current stored in the RL integrator from the last step (A).

    Always 0.0 for the degraded path.
    """
    return self._I_prev

  # ------------------------------------------------------------------ #
  # Private helpers                                                      #
  # ------------------------------------------------------------------ #

  def _full_rl_step(
      self,
      vel: float,
      effort_des: float,
      Kt: float,
      Ke: float,
      R: float,
      L: float,
      v_min: float,
      v_max: float,
      peak_torque: float,
      dt: float,
  ) -> tuple[float, float]:
    """Full RL-circuit integration step."""
    back_emf = Ke * vel
    I_target = effort_des / Kt
    # Desired terminal voltage (Euler discretisation of L dI/dt).
    V_terminal = I_target * R + L * (I_target - self._I_prev) / dt + back_emf
    # Clip to supply rail.
    voltage = float(np.clip(V_terminal, v_min, v_max))
    # Actual current delivered at the clipped voltage.
    I_actual = (voltage - back_emf + L * self._I_prev / dt) / (R + L / dt)
    torque = float(np.clip(Kt * I_actual, -peak_torque, peak_torque))
    return torque, float(I_actual)

  def _degraded_step(
      self,
      effort_des: float,
      Kt: float,
      peak_torque: float,
  ) -> tuple[float, float]:
    """Degraded path: clamp effort, approximate current from Kt."""
    torque = float(np.clip(effort_des, -peak_torque, peak_torque))
    current = torque / Kt
    return torque, current

  def _update_thermal(self, current: float, dt: float) -> None:
    """First-order thermal model: τ dT/dt = I²R − (T − T_amb)/R_th."""
    s = self.spec
    tau = s.thermal_time_constant   # type: ignore[assignment]
    R_th = s.thermal_resistance     # type: ignore[assignment]
    R = s.resistance                # type: ignore[assignment]
    dT = (current ** 2 * R - (self._temp - s.ambient_temperature) / R_th) / tau
    self._temp += dT * dt
