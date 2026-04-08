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
"""Read and write motor/battery specification tags in MuJoCo XML.

Conventions
-----------
Motor reference inside ``<custom>``::

    <text name="motor_{actuator_name}" data="motor_spec:{motor_id}"/>

Battery reference inside ``<custom>``::

    <text name="battery_{label}" data="battery_spec:{battery_id}"/>

Example::

    <mujoco>
      <custom>
        <text name="motor_a0"     data="motor_spec:faulhaber_2264w024bp4"/>
        <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
      </custom>
    </mujoco>

The functions in this module operate on :class:`mujoco.MjSpec` objects (the
Python-level scene-spec API).  ``parse_*`` functions return dictionaries that
map logical names to spec IDs; ``write_*`` functions add ``<text>`` children
to the ``custom`` block of the provided spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import mujoco

_MOTOR_NAME_PREFIX = "motor_"
_MOTOR_DATA_PREFIX = "motor_spec:"
_BATTERY_NAME_PREFIX = "battery_"
_BATTERY_DATA_PREFIX = "battery_spec:"


# --------------------------------------------------------------------------- #
# Parse helpers                                                                 #
# --------------------------------------------------------------------------- #

def parse_motor_specs_from_xml(spec: "mujoco.MjSpec") -> dict[str, str]:
  """Scan *spec* for motor-spec tags and return ``{actuator_name: motor_id}``.

  Parameters
  ----------
  spec:
      A :class:`mujoco.MjSpec` instance (loaded or constructed before
      compilation).

  Returns
  -------
  dict[str, str]
      Keys are actuator names (the part of the ``name`` attribute after
      ``"motor_"``).  Values are motor IDs (the part of the ``data``
      attribute after ``"motor_spec:"``).
  """
  result: dict[str, str] = {}
  for text in spec.texts:
    name: str = text.name
    data: str = text.data
    if name.startswith(_MOTOR_NAME_PREFIX) and data.startswith(
        _MOTOR_DATA_PREFIX
    ):
      actuator_name = name[len(_MOTOR_NAME_PREFIX):]
      motor_id = data[len(_MOTOR_DATA_PREFIX):]
      result[actuator_name] = motor_id
  return result


def parse_battery_specs_from_xml(spec: "mujoco.MjSpec") -> dict[str, str]:
  """Scan *spec* for battery-spec tags and return ``{label: battery_id}``.

  Parameters
  ----------
  spec:
      A :class:`mujoco.MjSpec` instance.

  Returns
  -------
  dict[str, str]
      Keys are battery labels (the part of the ``name`` attribute after
      ``"battery_"``).  Values are battery IDs (the part of the ``data``
      attribute after ``"battery_spec:"``).
  """
  result: dict[str, str] = {}
  for text in spec.texts:
    name: str = text.name
    data: str = text.data
    if name.startswith(_BATTERY_NAME_PREFIX) and data.startswith(
        _BATTERY_DATA_PREFIX
    ):
      label = name[len(_BATTERY_NAME_PREFIX):]
      battery_id = data[len(_BATTERY_DATA_PREFIX):]
      result[label] = battery_id
  return result


# --------------------------------------------------------------------------- #
# Write helpers                                                                 #
# --------------------------------------------------------------------------- #

def write_motor_spec_to_xml(
    spec: "mujoco.MjSpec",
    actuator_name: str,
    motor_id: str,
) -> None:
  """Add a motor-spec ``<text>`` tag to *spec*.

  Parameters
  ----------
  spec:
      A :class:`mujoco.MjSpec` instance to modify in place.
  actuator_name:
      Name of the MuJoCo actuator that uses this motor.
  motor_id:
      Unique motor identifier, e.g. ``'faulhaber_2264w024bp4'``.
  """
  text = spec.add_text()
  text.name = f"{_MOTOR_NAME_PREFIX}{actuator_name}"
  text.data = f"{_MOTOR_DATA_PREFIX}{motor_id}"


def write_battery_spec_to_xml(
    spec: "mujoco.MjSpec",
    label: str,
    battery_id: str,
) -> None:
  """Add a battery-spec ``<text>`` tag to *spec*.

  Parameters
  ----------
  spec:
      A :class:`mujoco.MjSpec` instance to modify in place.
  label:
      Logical battery label, e.g. ``'main'``.
  battery_id:
      Unique battery identifier, e.g. ``'unitree_g1_9ah'``.
  """
  text = spec.add_text()
  text.name = f"{_BATTERY_NAME_PREFIX}{label}"
  text.data = f"{_BATTERY_DATA_PREFIX}{battery_id}"
