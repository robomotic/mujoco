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
"""Electrical motor and battery simulation for MuJoCo (pure NumPy)."""

from mujoco.electrical.battery_manager import BatteryManager
from mujoco.electrical.battery_spec import BatterySpecification
from mujoco.electrical.database import BatteryDatabase
from mujoco.electrical.database import MotorDatabase
from mujoco.electrical.electrical_motor import ElectricalMotor
from mujoco.electrical.motor_spec import MotorSpecification
from mujoco.electrical.simulation import SimulationState
from mujoco.electrical.simulation import SingleEnvSimulation
from mujoco.electrical.xml_integration import parse_battery_specs_from_xml
from mujoco.electrical.xml_integration import parse_motor_specs_from_xml
from mujoco.electrical.xml_integration import write_battery_spec_to_xml
from mujoco.electrical.xml_integration import write_motor_spec_to_xml

__all__ = [
    "BatteryDatabase",
    "BatteryManager",
    "BatterySpecification",
    "ElectricalMotor",
    "MotorDatabase",
    "MotorSpecification",
    "SimulationState",
    "SingleEnvSimulation",
    "parse_battery_specs_from_xml",
    "parse_motor_specs_from_xml",
    "write_battery_spec_to_xml",
    "write_motor_spec_to_xml",
]
