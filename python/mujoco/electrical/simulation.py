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
"""Single-environment electrical motor + battery simulation.

:class:`SingleEnvSimulation` wires one :class:`~mujoco.electrical.battery_manager.BatteryManager`
to *N* :class:`~mujoco.electrical.electrical_motor.ElectricalMotor` instances
and integrates them with a compiled :class:`mujoco.MjModel`.

Integration order each step (matches mjlab ``scene.py``)
---------------------------------------------------------
1. Read zero-current bus voltage from battery (no state change).
2. For each actuator: ``motor.compute_control(…, bus_voltage=bus_voltage)``
   → torque, current.  Set ``data.ctrl[adr] = torque``.
3. Advance battery: ``battery.step(ΣI, dt)`` → terminal voltage under load.
4. Physics step: ``mujoco.mj_step(model, data)``.

Motor winding temperatures are updated inside
:meth:`~mujoco.electrical.electrical_motor.ElectricalMotor.compute_control`
(step 2), so no separate post-update is required.

XML auto-discovery
------------------
Motor and battery spec IDs are read from ``<custom><text>`` elements in the
model XML at construction time using plain :mod:`xml.etree.ElementTree`
(no MuJoCo C extension required for the discovery phase).  The convention
mirrors ``mjlab``:

.. code-block:: xml

    <custom>
      <text name="motor_a0"     data="motor_spec:faulhaber_2264w024bp4"/>
      <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
    </custom>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from mujoco.electrical.battery_manager import BatteryManager
from mujoco.electrical.battery_spec import BatterySpecification
from mujoco.electrical.electrical_motor import ElectricalMotor
from mujoco.electrical.motor_spec import MotorSpecification


# --------------------------------------------------------------------------- #
# SimulationState                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class SimulationState:
  """Snapshot of simulation state returned by :meth:`SingleEnvSimulation.step`.

  All arrays are copies — modifying them does not affect the simulation.
  """

  qpos: np.ndarray
  """Joint positions at the *end* of the step (rad)."""

  qvel: np.ndarray
  """Joint velocities at the *end* of the step (rad/s)."""

  torques: np.ndarray
  """Torques applied to each actuator during the step (N·m).  Shape: (n,)."""

  currents: np.ndarray
  """Winding currents for each motor (A).  Shape: (n,)."""

  soc: float
  """Battery state of charge in [0, 1] after the step.  ``nan`` if no battery."""

  bus_voltage: float
  """Terminal voltage under load returned by :meth:`BatteryManager.step` (V).
  ``nan`` if no battery.
  """

  temperatures: np.ndarray
  """Winding temperatures for each motor (°C).  Shape: (n,)."""

  time: float = 0.0
  """Simulation time at the *end* of the step (s)."""


# --------------------------------------------------------------------------- #
# SingleEnvSimulation                                                           #
# --------------------------------------------------------------------------- #

class SingleEnvSimulation:
  """Single-environment electrical motor + battery simulation.

  Prefer constructing via :meth:`from_xml` which handles auto-discovery.
  Direct construction is also supported for programmatic use.

  Args:
    model:         Compiled :class:`mujoco.MjModel`.
    data:          :class:`mujoco.MjData` associated with *model*.
    motors:        Ordered list of :class:`ElectricalMotor` instances.
    battery:       :class:`BatteryManager`, or ``None`` for an unmetered supply.
    actuator_adrs: MuJoCo actuator indices corresponding to each motor.
  """

  def __init__(
      self,
      model,
      data,
      motors: list[ElectricalMotor],
      battery: BatteryManager | None,
      actuator_adrs: list[int],
  ) -> None:
    if len(motors) != len(actuator_adrs):
      raise ValueError(
          f"motors ({len(motors)}) and actuator_adrs ({len(actuator_adrs)}) "
          "must have the same length"
      )
    self._model = model
    self._data = data
    self._motors = motors
    self._battery = battery
    self._actuator_adrs = actuator_adrs

  # ------------------------------------------------------------------ #
  # Construction                                                         #
  # ------------------------------------------------------------------ #

  @classmethod
  def from_xml(
      cls,
      xml_string: str,
      motor_db=None,
      battery_db=None,
      kp: float = 100.0,
      kd: float = 10.0,
  ) -> "SingleEnvSimulation":
    """Build a simulation from a MuJoCo XML string.

    Motor and battery spec IDs are discovered automatically from
    ``<custom><text>`` tags.  Missing ``motor_db`` / ``battery_db``
    arguments fall back to :class:`~mujoco.electrical.database.MotorDatabase`
    and :class:`~mujoco.electrical.database.BatteryDatabase` with default
    remote sources.

    Args:
      xml_string:  Full MuJoCo XML model string.
      motor_db:    Optional :class:`~mujoco.electrical.database.MotorDatabase`.
      battery_db:  Optional :class:`~mujoco.electrical.database.BatteryDatabase`.
      kp:          Position-error proportional gain for all motors (N·m/rad).
      kd:          Velocity-error derivative gain for all motors (N·m·s/rad).

    Returns:
      Fully initialised :class:`SingleEnvSimulation`.

    Raises:
      ImportError: If the MuJoCo C extension is not installed.
      ValueError:  If a motor references an actuator not found in the model.
    """
    import mujoco  # noqa: PLC0415 — requires C extension at runtime
    from mujoco.electrical.database import BatteryDatabase  # noqa: PLC0415
    from mujoco.electrical.database import MotorDatabase  # noqa: PLC0415

    if motor_db is None:
      motor_db = MotorDatabase()
    if battery_db is None:
      battery_db = BatteryDatabase()

    # Phase 1: discover spec IDs from raw XML (no C extension needed).
    motor_map, battery_map = cls._discover_specs(xml_string)

    # Phase 2: compile MuJoCo model.
    # MjSpec.from_string is a classmethod in the dev build but an instance
    # method in the released 3.x packages — handle both.
    try:
      spec = mujoco.MjSpec.from_string(xml_string)
    except TypeError:
      spec = mujoco.MjSpec()
      spec.from_string(xml_string)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Phase 3: resolve actuator indices and load motor specs.
    motors: list[ElectricalMotor] = []
    actuator_adrs: list[int] = []
    for actuator_name, motor_id in motor_map.items():
      adr = mujoco.mj_name2id(
          model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
      )
      if adr == -1:
        raise ValueError(
            f"Actuator '{actuator_name}' referenced in motor_spec tag "
            "was not found in the compiled model."
        )
      motor_spec: MotorSpecification = motor_db.load(motor_id)
      motors.append(ElectricalMotor(motor_spec, kp=kp, kd=kd))
      actuator_adrs.append(adr)

    # Phase 4: load battery (first one found; single-env has one bus).
    battery: BatteryManager | None = None
    for battery_id in battery_map.values():
      battery_spec: BatterySpecification = battery_db.load(battery_id)
      battery = BatteryManager(battery_spec)
      break  # single environment — one battery bus

    return cls(model, data, motors, battery, actuator_adrs)

  # ------------------------------------------------------------------ #
  # Simulation                                                           #
  # ------------------------------------------------------------------ #

  def step(
      self,
      pos_targets: np.ndarray,
      vel_targets: np.ndarray | None = None,
      effort_targets: np.ndarray | None = None,
  ) -> SimulationState:
    """Advance the simulation by one time step.

    Args:
      pos_targets:    Desired actuator positions (rad).  Shape: (n,).
      vel_targets:    Desired actuator velocities (rad/s).  Defaults to zeros.
      effort_targets: Feed-forward torques (N·m).  Defaults to zeros.

    Returns:
      :class:`SimulationState` snapshot after the physics step.
    """
    import mujoco  # noqa: PLC0415

    n = len(self._motors)
    pos_targets = np.asarray(pos_targets, dtype=float)
    vel_targets = (
        np.zeros(n, dtype=float)
        if vel_targets is None
        else np.asarray(vel_targets, dtype=float)
    )
    effort_targets = (
        np.zeros(n, dtype=float)
        if effort_targets is None
        else np.asarray(effort_targets, dtype=float)
    )

    dt: float = float(self._model.opt.timestep)
    torques = np.zeros(n)
    currents = np.zeros(n)

    # 1. Zero-current bus voltage (non-destructive read).
    bus_voltage: float | None = (
        self._battery.terminal_voltage if self._battery is not None else None
    )

    # 2. Compute motor controls and write ctrl array.
    for i, (motor, adr) in enumerate(zip(self._motors, self._actuator_adrs)):
      pos = float(self._data.actuator_length[adr])
      vel = float(self._data.actuator_velocity[adr])
      torque, current = motor.compute_control(
          pos=pos,
          vel=vel,
          pos_tgt=float(pos_targets[i]),
          vel_tgt=float(vel_targets[i]),
          effort_tgt=float(effort_targets[i]),
          dt=dt,
          bus_voltage=bus_voltage,
      )
      torques[i] = torque
      currents[i] = current
      self._data.ctrl[adr] = torque

    # 3. Advance battery with aggregate current; get terminal voltage under load.
    if self._battery is not None:
      bus_voltage = self._battery.step(float(np.sum(currents)), dt)

    # 4. Physics step.
    mujoco.mj_step(self._model, self._data)

    temps = np.array([m.winding_temperature for m in self._motors])

    return SimulationState(
        qpos=self._data.qpos.copy(),
        qvel=self._data.qvel.copy(),
        torques=torques,
        currents=currents,
        soc=(self._battery.soc if self._battery is not None else float("nan")),
        bus_voltage=(
            bus_voltage if bus_voltage is not None else float("nan")
        ),
        temperatures=temps,
        time=float(self._data.time),
    )

  def reset(self, initial_soc: float = 1.0) -> None:
    """Reset physics state, motor integrators, and battery.

    Args:
      initial_soc: Battery SOC to reset to (default 1.0 = full).
    """
    import mujoco  # noqa: PLC0415

    mujoco.mj_resetData(self._model, self._data)
    for motor in self._motors:
      motor.reset()
    if self._battery is not None:
      self._battery.reset(initial_soc)

  # ------------------------------------------------------------------ #
  # Properties                                                           #
  # ------------------------------------------------------------------ #

  @property
  def model(self):
    """The compiled :class:`mujoco.MjModel`."""
    return self._model

  @property
  def data(self):
    """The :class:`mujoco.MjData` associated with :attr:`model`."""
    return self._data

  @property
  def n_actuators(self) -> int:
    """Number of electrically-modelled actuators."""
    return len(self._motors)

  @property
  def motors(self) -> list[ElectricalMotor]:
    """Ordered list of :class:`ElectricalMotor` instances."""
    return self._motors

  @property
  def battery(self) -> BatteryManager | None:
    """The :class:`BatteryManager`, or ``None`` if none was configured."""
    return self._battery

  # ------------------------------------------------------------------ #
  # Internal helpers                                                     #
  # ------------------------------------------------------------------ #

  @staticmethod
  def _discover_specs(
      xml_string: str,
  ) -> tuple[dict[str, str], dict[str, str]]:
    """Extract motor and battery spec IDs from raw MuJoCo XML.

    Parses ``<custom><text name="motor_X" data="motor_spec:Y"/>`` and
    ``<custom><text name="battery_X" data="battery_spec:Y"/>`` elements
    using :mod:`xml.etree.ElementTree` — no MuJoCo C extension required.

    Args:
      xml_string: Raw MuJoCo XML model string.

    Returns:
      ``(motor_map, battery_map)`` where:

      * *motor_map* maps actuator name → motor ID.
      * *battery_map* maps label → battery ID.
    """
    root = ET.fromstring(xml_string)
    motor_map: dict[str, str] = {}
    battery_map: dict[str, str] = {}
    for text_el in root.findall(".//custom/text"):
      name: str = text_el.get("name", "")
      data: str = text_el.get("data", "")
      if name.startswith("motor_") and data.startswith("motor_spec:"):
        motor_map[name[len("motor_"):]] = data[len("motor_spec:"):]
      elif name.startswith("battery_") and data.startswith("battery_spec:"):
        battery_map[name[len("battery_"):]] = data[len("battery_spec:"):]
    return motor_map, battery_map
