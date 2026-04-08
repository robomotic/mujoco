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
"""Single-environment battery simulation.

The :class:`BatteryManager` advances pack state (SOC, temperature) and returns
the terminal voltage seen by the motor bus.  All state is scalar; the class
is designed for ``SingleEnvSimulation`` and is not batched.

Physics summary
---------------
Open-circuit voltage (per cell, then scaled to pack):

.. math::

   V_{oc} = \\text{interp}(\\text{SOC},\\; \\text{ocv\\_soc},\\; \\text{ocv\\_v})
             \\times N_s

Internal resistance (SOC- and temperature-dependent):

.. math::

   R_{mult} = \\text{interp}(\\text{SOC},\\; \\text{r\\_soc},\\; \\text{r\\_mult})

   R_{int} = R_0 \\times R_{mult} \\times (1 + k_{\\Delta T} \\cdot (T - T_{amb}))

Terminal voltage (clamped to pack limits):

.. math::

   V_{term} = \\text{clip}(V_{oc} - I \\cdot R_{int},\\; V_{min},\\; V_{max})

SOC update (Coulomb counting):

.. math::

   \\Delta \\text{SOC} = -\\frac{I \\cdot \\Delta t}{C_{Ah} \\times 3600}

Thermal update (Newton cooling + Joule heating):

.. math::

   \\Delta T = \\frac{I^2 R_{int} - (T - T_{amb}) / R_{th}}{C_{th}} \\cdot \\Delta t
"""

from __future__ import annotations

import numpy as np

from mujoco.electrical.battery_spec import BatterySpecification


class BatteryManager:
  """NumPy single-environment battery state manager.

  Args:
    spec:        Battery specification loaded from a JSON asset.
    initial_soc: Initial state of charge in [0, 1].  Defaults to 1.0 (full).
  """

  def __init__(
      self,
      spec: BatterySpecification,
      initial_soc: float = 1.0,
  ) -> None:
    self.spec = spec
    self._soc: float = float(np.clip(initial_soc, 0.0, 1.0))
    self._temp: float = float(spec.ambient_temperature)

  # ------------------------------------------------------------------ #
  # Public API                                                           #
  # ------------------------------------------------------------------ #

  def reset(self, initial_soc: float = 1.0) -> None:
    """Reset pack state for a new episode.

    Args:
      initial_soc: State of charge to reset to (default 1.0 = full).
    """
    self._soc = float(np.clip(initial_soc, 0.0, 1.0))
    self._temp = float(self.spec.ambient_temperature)

  def step(self, current_draw: float, dt: float) -> float:
    """Advance battery state by one time step.

    Args:
      current_draw: Total discharge current drawn by all motors (A).
                    Sign convention: positive = discharging.
      dt:           Simulation time step (s).  Must be > 0.

    Returns:
      Terminal voltage (V) at the start of this step, *before* the SOC
      and temperature are updated.  This is the voltage available to the
      motor bus during the step.

    Raises:
      ValueError: If *dt* is not positive.
    """
    if dt <= 0.0:
      raise ValueError(f"dt must be positive, got {dt!r}")

    s = self.spec
    v_term = self._terminal_voltage(current_draw)

    # SOC update — Coulomb counting, clamped to [0, 1]
    delta_soc = -current_draw / (s.capacity_ah * 3600.0) * dt
    self._soc = float(np.clip(self._soc + delta_soc, 0.0, 1.0))

    # Thermal update (only when thermal_capacity is finite / positive)
    if s.thermal_capacity > 0.0:
      r_int = self._internal_resistance(current_draw)
      joule_heat = current_draw ** 2 * r_int
      cooling = (self._temp - s.ambient_temperature) / s.thermal_resistance
      dT = (joule_heat - cooling) / s.thermal_capacity
      self._temp = float(self._temp + dT * dt)

    return v_term

  # ------------------------------------------------------------------ #
  # Read-only state properties                                           #
  # ------------------------------------------------------------------ #

  @property
  def soc(self) -> float:
    """State of charge in [0, 1]."""
    return self._soc

  @property
  def temperature(self) -> float:
    """Pack temperature (°C)."""
    return self._temp

  @property
  def terminal_voltage(self) -> float:
    """Current terminal voltage at zero current draw (V).

    Useful for polling the bus voltage without advancing state.
    Equivalent to calling ``step(0.0, dt)`` without side-effects.
    """
    return self._terminal_voltage(0.0)

  @property
  def is_depleted(self) -> bool:
    """True when SOC has reached the minimum safe level."""
    return self._soc <= self.spec.min_soc

  @property
  def is_overheated(self) -> bool:
    """True when pack temperature exceeds the maximum safe level."""
    return self._temp >= self.spec.max_temperature

  # ------------------------------------------------------------------ #
  # Private helpers                                                      #
  # ------------------------------------------------------------------ #

  def _ocv(self) -> float:
    """Open-circuit voltage (V) interpolated from the OCV curve."""
    s = self.spec
    v_cell = float(np.interp(self._soc, s.ocv_soc, s.ocv_v))
    return v_cell * s.cells_series

  def _internal_resistance(self, current_draw: float) -> float:  # noqa: ARG002
    """Effective internal resistance (Ω) at current SOC and temperature."""
    s = self.spec
    r_mult = float(np.interp(self._soc, s.r_soc, s.r_mult))
    temp_factor = 1.0 + s.internal_resistance_temp_coeff * (
        self._temp - s.ambient_temperature
    )
    return s.internal_resistance * r_mult * temp_factor

  def _terminal_voltage(self, current_draw: float) -> float:
    """Compute terminal voltage without advancing state."""
    s = self.spec
    v_oc = self._ocv()
    r_int = self._internal_resistance(current_draw)
    v_term = v_oc - current_draw * r_int
    return float(np.clip(v_term, s.min_voltage, s.max_voltage))
