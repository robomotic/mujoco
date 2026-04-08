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
"""CLI tool for downloading and caching motor / battery specifications.

Usage examples::

    # Download a fully-specified motor (default / recommended)
    python -m mujoco.electrical.scripts.download_specs motor faulhaber_2264w024bp4

    # Download a battery spec
    python -m mujoco.electrical.scripts.download_specs battery unitree_g1_9ah

    # List all spec IDs available in ~/.mujoco/cache/
    python -m mujoco.electrical.scripts.download_specs --list-cached motor
    python -m mujoco.electrical.scripts.download_specs --list-cached battery

    # Show remote URL that would be used (dry-run)
    python -m mujoco.electrical.scripts.download_specs --dry-run motor faulhaber_2264w024bp4
"""

from __future__ import annotations

import argparse
import json
import sys

from mujoco.electrical.database import (
    BatteryDatabase,
    MotorDatabase,
    _BATTERY_REMOTE_BASE,
    _MOTOR_REMOTE_BASE,
    _vendor_from_id,
)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="python -m mujoco.electrical.scripts.download_specs",
      description="Download and cache motor or battery specifications.",
  )
  parser.add_argument(
      "kind",
      choices=["motor", "battery"],
      help="Type of specification to download.",
  )
  parser.add_argument(
      "spec_id",
      nargs="?",
      default=None,
      help=(
          "Unique spec identifier, e.g. 'faulhaber_2264w024bp4' or "
          "'unitree_g1_9ah'.  Required unless --list-cached is used."
      ),
  )
  parser.add_argument(
      "--list-cached",
      action="store_true",
      help="List cached spec IDs and exit.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print the remote URL that would be fetched, then exit.",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Re-download even if a cached copy already exists.",
  )
  parser.add_argument(
      "--path",
      default=None,
      metavar="FILE",
      help="Load from an explicit local JSON file instead of remote.",
  )
  return parser


def _remote_url(kind: str, spec_id: str) -> str:
  vendor = _vendor_from_id(spec_id)
  if kind == "motor":
    return f"{_MOTOR_REMOTE_BASE}/{vendor}/{spec_id}.json"
  return f"{_BATTERY_REMOTE_BASE}/{vendor}/{spec_id}.json"


def main(argv: list[str] | None = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)

  if args.list_cached:
    if args.kind == "motor":
      ids = MotorDatabase().list_cached()
    else:
      ids = BatteryDatabase().list_cached()
    if ids:
      print("\n".join(sorted(ids)))
    else:
      print(f"No cached {args.kind} specs found in ~/.mujoco/cache/.")
    return 0

  if args.spec_id is None:
    parser.error("spec_id is required unless --list-cached is used.")

  if args.dry_run:
    url = _remote_url(args.kind, args.spec_id)
    print(f"Would fetch: {url}")
    return 0

  try:
    if args.kind == "motor":
      spec = MotorDatabase().load(args.spec_id, path=args.path)
      print(
          f"Loaded motor: {spec.motor_id}"
          f" ({spec.manufacturer} {spec.model})"
      )
      print(f"  RL circuit : {'yes' if spec.has_rl_circuit else 'no (degraded path)'}")
      print(f"  Thermal    : {'yes' if spec.has_thermal else 'no'}")
      print(f"  Peak torque: {spec.peak_torque} N·m")
      print(f"  Voltage    : {spec.voltage_range[0]}–{spec.voltage_range[1]} V")
    else:
      spec = BatteryDatabase().load(args.spec_id, path=args.path)
      print(
          f"Loaded battery: {spec.battery_id}"
          f" ({spec.manufacturer} {spec.model})"
      )
      print(f"  Chemistry  : {spec.chemistry}")
      print(f"  Config     : {spec.cells_series}S{spec.cells_parallel}P")
      print(f"  Capacity   : {spec.capacity_ah} Ah")
      print(f"  Voltage    : {spec.nominal_voltage:.1f} V nominal")
      print(f"  Energy     : {spec.energy_wh:.1f} Wh")
  except FileNotFoundError as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1

  return 0


if __name__ == "__main__":
  sys.exit(main())
