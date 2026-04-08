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
"""Motor specification dataclass.

The canonical fully-specified motor is ``faulhaber_2264w024bp4``.  Some
community specs (e.g. ``unitree_a1``) have ``None`` for the RL-circuit and
thermal fields; the simulation handles both cases gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Optional


@dataclass
class MotorSpecification:
  """Electrical motor specification.

  Required fields must be present in every JSON asset.  Optional fields
  may be ``null`` / absent; the ``has_rl_circuit`` and ``has_thermal``
  properties indicate which physics paths are available at runtime.

  Voltage / current / torque values use SI units (V, A, N·m, rad/s).
  """

  # ------------------------------------------------------------------ #
  # Identity                                                             #
  # ------------------------------------------------------------------ #
  motor_id: str
  """Unique identifier matching the JSON filename, e.g. 'faulhaber_2264w024bp4'."""

  manufacturer: str
  """Manufacturer name, e.g. 'Faulhaber'."""

  model: str
  """Part / model number, e.g. '2264W024BP4'."""

  # ------------------------------------------------------------------ #
  # Required electrical parameters                                       #
  # ------------------------------------------------------------------ #
  gear_ratio: float
  """Gear ratio (output / motor).  1.0 = direct drive."""

  motor_constant_kt: float
  """Torque constant Kt (N·m/A)."""

  motor_constant_ke: float
  """Back-EMF constant Ke (V·s/rad)."""

  peak_torque: float
  """Peak instantaneous torque at the output shaft (N·m)."""

  stall_torque: float
  """Stall torque (N·m)."""

  continuous_torque: float
  """Continuous rated torque (N·m)."""

  no_load_speed: float
  """No-load angular speed (rad/s)."""

  no_load_current: float
  """No-load current draw (A)."""

  voltage_range: list[float]
  """[v_min, v_max] — operating voltage range (V)."""

  # ------------------------------------------------------------------ #
  # Optional — RL circuit (null in some real specs, e.g. unitree_a1)    #
  # ------------------------------------------------------------------ #
  resistance: Optional[float] = None
  """Winding resistance R (Ω).  None → degraded torque-clamp path."""

  inductance: Optional[float] = None
  """Winding inductance L (H).  None → degraded torque-clamp path."""

  # ------------------------------------------------------------------ #
  # Optional — thermal model                                             #
  # ------------------------------------------------------------------ #
  thermal_resistance: Optional[float] = None
  """Thermal resistance winding-to-ambient (°C/W)."""

  thermal_time_constant: Optional[float] = None
  """Thermal time constant τ_th (s)."""

  max_winding_temperature: Optional[float] = None
  """Maximum allowable winding temperature (°C)."""

  # ------------------------------------------------------------------ #
  # Extended optional fields (Faulhaber has all of these)               #
  # ------------------------------------------------------------------ #
  reflected_inertia: Optional[float] = None
  """Rotor inertia reflected to motor shaft (kg·m²)."""

  number_of_pole_pairs: Optional[int] = None
  """Number of magnetic pole pairs."""

  max_speed: Optional[float] = None
  """Maximum continuous speed (rad/s)."""

  weight: Optional[float] = None
  """Motor mass (kg)."""

  commutation: Optional[str] = None
  """Commutation type, e.g. 'Hall', 'Encoder', 'Brush'."""

  friction_static: Optional[float] = None
  """Static friction torque (N·m)."""

  friction_dynamic: Optional[float] = None
  """Dynamic (viscous) friction coefficient (N·m·s/rad)."""

  rotation_angle_range: Optional[list[float]] = None
  """[min_rad, max_rad] joint angle limits.  [0,0] = continuous."""

  stall_current: Optional[float] = None
  """Stall current (A)."""

  operating_current: Optional[float] = None
  """Rated operating current (A)."""

  ambient_temperature: float = 25.0
  """Ambient / initial winding temperature (°C)."""

  encoder_resolution: Optional[int] = None
  """Encoder resolution (counts/rev)."""

  encoder_type: Optional[str] = None
  """Encoder type, e.g. 'incremental', 'absolute'."""

  protocol: Optional[str] = None
  """Communication protocol, e.g. 'PWM', 'RS-485', 'CANopen'."""

  step_file: Optional[str] = None
  """URL to STEP CAD file."""

  stl_file: Optional[str] = None
  """URL to STL mesh file."""

  # ------------------------------------------------------------------ #
  # Derived helpers                                                      #
  # ------------------------------------------------------------------ #
  @property
  def has_rl_circuit(self) -> bool:
    """True when both resistance and inductance are specified."""
    return self.resistance is not None and self.inductance is not None

  @property
  def has_thermal(self) -> bool:
    """True when thermal resistance and time constant are specified."""
    return (
        self.thermal_resistance is not None
        and self.thermal_time_constant is not None
    )

  @classmethod
  def from_dict(cls, data: dict) -> "MotorSpecification":
    """Construct from a JSON-parsed dictionary.

    Unknown keys are ignored so that future spec extensions are
    forwards-compatible.
    """
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)
