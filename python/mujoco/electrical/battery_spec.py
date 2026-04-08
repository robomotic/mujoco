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
"""Battery specification dataclass.

Supports common robotics battery chemistries: LiPo, LiFePO4, Li-ion.
The reference spec is ``unitree_g1_9ah`` (6S LiPo, 9 Ah).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Optional


@dataclass
class BatterySpecification:
  """Battery pack specification with electrical, thermal, and chemical properties.

  All voltage values are *per-cell* in the JSON asset; pack voltages are
  computed by multiplying by ``cells_series`` at load time and stored in the
  ``nominal_voltage``, ``max_voltage``, ``min_voltage`` fields.

  The OCV and resistance-SOC curves are lists of ``[soc, value]`` pairs
  sorted by SOC (0.0 → 1.0).
  """

  # ------------------------------------------------------------------ #
  # Identity                                                             #
  # ------------------------------------------------------------------ #
  battery_id: str
  """Unique identifier matching the JSON filename, e.g. 'unitree_g1_9ah'."""

  manufacturer: str
  """Manufacturer name, e.g. 'Unitree'."""

  model: str
  """Model designation, e.g. 'G1-9Ah'."""

  # ------------------------------------------------------------------ #
  # Chemistry & configuration                                            #
  # ------------------------------------------------------------------ #
  chemistry: str
  """Battery chemistry: 'LiPo', 'LiFePO4', or 'Li-ion'."""

  cells_series: int
  """Number of cells in series (e.g. 6 for a 6S pack)."""

  cells_parallel: int
  """Number of cells in parallel (e.g. 1 for a 1P pack)."""

  nominal_cell_voltage: float
  """Nominal voltage per cell (V): 3.7 (LiPo), 3.2 (LiFe), 3.6 (Li-ion)."""

  max_cell_voltage: float
  """Maximum cell voltage (V): typically 4.2 (LiPo) or 3.65 (LiFe)."""

  min_cell_voltage: float
  """Minimum cell voltage (V): typically 3.0 (LiPo) or 2.5 (LiFe)."""

  # ------------------------------------------------------------------ #
  # Capacity & energy                                                    #
  # ------------------------------------------------------------------ #
  capacity_ah: float
  """Pack capacity in amp-hours (Ah)."""

  # ------------------------------------------------------------------ #
  # Electrical properties                                                #
  # ------------------------------------------------------------------ #
  internal_resistance: float
  """Pack internal resistance at nominal SOC and temperature (Ω)."""

  max_continuous_current: float
  """Maximum continuous discharge current (A)."""

  # ------------------------------------------------------------------ #
  # Optional electrical properties                                       #
  # ------------------------------------------------------------------ #
  min_operating_voltage: Optional[float] = None
  """Conservative per-cell cutoff voltage (V) to prevent deep discharge."""

  internal_resistance_temp_coeff: float = 0.0
  """Temperature coefficient of resistance (Ω/°C)."""

  internal_resistance_soc_curve: Optional[list[list[float]]] = None
  """SOC-dependent resistance multiplier [[soc, multiplier], ...].
  Example: [[0.0, 2.5], [0.2, 1.5], [0.5, 1.0], [1.0, 1.0]].
  If None a default curve is used (2.5× at empty → 1.0× at full).
  """

  max_burst_current: Optional[float] = None
  """Maximum burst discharge current (A).  Defaults to 2× continuous."""

  burst_duration: float = 10.0
  """Maximum burst duration (s)."""

  # ------------------------------------------------------------------ #
  # Discharge curve                                                      #
  # ------------------------------------------------------------------ #
  ocv_curve: Optional[list[list[float]]] = None
  """Open-circuit voltage curve [[soc, v_per_cell], ...].
  If None a linear curve between min and max cell voltage is used.
  """

  # ------------------------------------------------------------------ #
  # Thermal properties                                                   #
  # ------------------------------------------------------------------ #
  thermal_capacity: float = 1000.0
  """Thermal (heat) capacity of the pack (J/°C)."""

  thermal_resistance: float = 10.0
  """Thermal resistance pack-to-ambient (°C/W)."""

  max_temperature: float = 60.0
  """Maximum safe operating temperature (°C)."""

  min_temperature: float = 0.0
  """Minimum operating temperature (°C)."""

  ambient_temperature: float = 25.0
  """Initial / ambient temperature (°C)."""

  # ------------------------------------------------------------------ #
  # Depth-of-discharge limits                                            #
  # ------------------------------------------------------------------ #
  min_soc: float = 0.2
  """Minimum state of charge (0–1); typically 0.2 to prevent damage."""

  max_soc: float = 1.0
  """Maximum state of charge (0–1); typically 1.0."""

  # ------------------------------------------------------------------ #
  # Physical properties                                                  #
  # ------------------------------------------------------------------ #
  mass_kg: Optional[float] = None
  """Pack mass (kg)."""

  volume_liters: Optional[float] = None
  """Pack volume (L)."""

  # ------------------------------------------------------------------ #
  # Cell balancing                                                       #
  # ------------------------------------------------------------------ #
  cell_balance_tolerance: float = 0.05
  """Maximum acceptable inter-cell voltage imbalance (V)."""

  # ------------------------------------------------------------------ #
  # Derived pack voltages (set by __post_init__)                         #
  # ------------------------------------------------------------------ #
  nominal_voltage: float = field(init=False)
  """Pack nominal voltage = nominal_cell_voltage * cells_series (V)."""

  max_voltage: float = field(init=False)
  """Pack maximum voltage = max_cell_voltage * cells_series (V)."""

  min_voltage: float = field(init=False)
  """Pack minimum voltage = min_cell_voltage * cells_series (V)."""

  energy_wh: float = field(init=False)
  """Total energy capacity = capacity_ah * nominal_voltage (Wh)."""

  def __post_init__(self) -> None:
    self.nominal_voltage = self.nominal_cell_voltage * self.cells_series
    self.max_voltage = self.max_cell_voltage * self.cells_series
    self.min_voltage = self.min_cell_voltage * self.cells_series
    self.energy_wh = self.capacity_ah * self.nominal_voltage

    if self.max_burst_current is None:
      self.max_burst_current = self.max_continuous_current * 2.0

    if self.ocv_curve is None:
      self.ocv_curve = [
          [0.0, self.min_cell_voltage],
          [0.5, self.nominal_cell_voltage],
          [1.0, self.max_cell_voltage],
      ]

    if self.internal_resistance_soc_curve is None:
      self.internal_resistance_soc_curve = [
          [0.0, 2.5],
          [0.2, 1.5],
          [0.5, 1.0],
          [1.0, 1.0],
      ]

  # ------------------------------------------------------------------ #
  # Convenience accessors for the physics engine                         #
  # ------------------------------------------------------------------ #
  @property
  def ocv_soc(self) -> list[float]:
    """SOC breakpoints of the OCV curve."""
    return [pt[0] for pt in self.ocv_curve]  # type: ignore[index]

  @property
  def ocv_v(self) -> list[float]:
    """Per-cell voltage breakpoints of the OCV curve."""
    return [pt[1] for pt in self.ocv_curve]  # type: ignore[index]

  @property
  def r_soc(self) -> list[float]:
    """SOC breakpoints of the internal-resistance multiplier curve."""
    return [pt[0] for pt in self.internal_resistance_soc_curve]  # type: ignore[index]

  @property
  def r_mult(self) -> list[float]:
    """Resistance multiplier breakpoints."""
    return [pt[1] for pt in self.internal_resistance_soc_curve]  # type: ignore[index]

  @classmethod
  def from_dict(cls, data: dict) -> "BatterySpecification":
    """Construct from a JSON-parsed dictionary.

    Unknown keys are ignored so that future spec extensions are
    forwards-compatible.  Derived fields (``nominal_voltage`` etc.) are
    excluded from the constructor because they are set by ``__post_init__``.
    """
    _derived = {"nominal_voltage", "max_voltage", "min_voltage", "energy_wh"}
    known = {
        f.name
        for f in cls.__dataclass_fields__.values()  # type: ignore[attr-defined]
        if f.name not in _derived
    }
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)
