#!/usr/bin/env python3
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
"""GUI demo: live electrical metrics on an Acrobot-style MuJoCo model.

Opens the standard MuJoCo viewer window (via ``mujoco.viewer.launch_passive``)
and streams live electrical metrics into the viewer's built-in **Sensors**
panel while an **Acrobot-inspired** two-link mechanism runs.

The scene is based on ``dm_control/suite/acrobot.xml``: the shoulder joint is
passive and the elbow joint is driven through ``mujoco.electrical``. The
demo asks the elbow motor to hold a 90° bend, comparing a light tip payload
against a clearly overloaded heavy payload.

How the metrics appear
----------------------
Five ``<sensor type="user">`` elements are declared in the model XML::

    motor_current   [A]     cutoff = 120 A
    bus_voltage     [V]     cutoff =  30 V
    elec_power_w    [W]     cutoff = 3000 W
    battery_soc     [0-1]   cutoff =   1
    winding_temp_c  [°C]    cutoff = 125 °C

After each physics step the demo writes the computed electrical values into
``mjData.sensordata`` for those sensor slots, then calls ``handle.sync()``.
Press **F4** in the viewer to open the Sensor panel. The bars scale to each
sensor's ``cutoff`` value so you can see relative load at a glance.

If the installed MuJoCo build exposes ``Handle.set_texts()`` (available in
the repo build but not the 3.2.x binary release), a HUD overlay is also shown
in the top-right corner with a formatted metrics table.

Usage
-----
::

    # Run both payload scenarios back-to-back (default)
    PYTHONPATH=python python3 python/examples/electrical/demo_visualizer.py

    # Run one scenario only
    PYTHONPATH=python python3 python/examples/electrical/demo_visualizer.py --light
    PYTHONPATH=python python3 python/examples/electrical/demo_visualizer.py --heavy

    # Change simulation length
    PYTHONPATH=python python3 python/examples/electrical/demo_visualizer.py --steps 5000

Controls
--------
* Press **F4** inside the viewer to toggle the Sensors panel.
* Press **N** to skip to the next scenario.
* Close the window (×) or press **Esc** to exit.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Guard: require mujoco C extension before anything else.
# ---------------------------------------------------------------------------
try:
    import mujoco
    if not hasattr(mujoco, "MjSpec"):
        raise ImportError("MjSpec not found")
except ImportError:
    print(
        "ERROR: The MuJoCo C extension is not installed.\n"
        "Install it with:  pip install mujoco\n"
        "Then re-run this demo.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import mujoco.viewer
except ImportError:
    print(
        "ERROR: mujoco.viewer is not available.\n"
        "This demo requires the full MuJoCo Python package with GLFW support.",
        file=sys.stderr,
    )
    sys.exit(1)

from mujoco.electrical import SingleEnvSimulation  # noqa: E402

# ---------------------------------------------------------------------------
# Detect whether Handle.set_texts() is available in this build.
# It is defined in the repo's python/mujoco/simulate.cc but not in the
# upstream 3.2.x binary release.
# ---------------------------------------------------------------------------
_HAS_SET_TEXTS: bool = False
try:
    from mujoco._simulate import Simulate as _Simulate  # noqa: F401
    _HAS_SET_TEXTS = hasattr(_Simulate, "set_texts")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Scenario / XML — Acrobot-style model inspired by dm_control/suite/acrobot.xml
# ---------------------------------------------------------------------------
_LINK_LENGTH     = 1.0           # m
_UPPER_ARM_MASS  = 0.12          # kg
_LOWER_ARM_MASS  = 0.10          # kg
_GRAVITY         = 9.81          # m/s²
_TARGET_ELBOW    = math.pi / 2   # rad
_PEAK_TORQUE     = 1.311         # N·m  Faulhaber 2264W
_MAX_TEMP        = 125.0         # °C  Faulhaber winding limit

# Five user sensors expose electrical metrics to the built-in Sensors panel.
# cutoff = normalization full-scale for the sensor bar chart.
_XML_TEMPLATE = """\
<mujoco model="electrical_acrobot_demo">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <visual>
    <headlight diffuse="0.85 0.85 0.85" ambient="0.35 0.35 0.35"
               specular="0.20 0.20 0.20"/>
    <map zfar="20"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global elevation="-18" azimuth="135"/>
  </visual>

  <statistic center="0 0 0.8" extent="2.5"/>

  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.30 0.50 0.70" rgb2="0.02 0.04 0.08"
             width="512" height="512"/>
    <texture name="body" type="cube" builtin="flat" mark="cross"
             width="128" height="128"
             rgb1="0.75 0.78 0.82" rgb2="0.75 0.78 0.82"
             markrgb="1 1 1"/>
    <material name="body" texture="body" texuniform="true"
              rgba="0.82 0.84 0.88 1"/>
    <texture name="grid" type="2d" builtin="checker"
             width="512" height="512"
             rgb1="0.10 0.20 0.30" rgb2="0.20 0.30 0.40"/>
    <material name="grid" texture="grid" texrepeat="4 4"
              texuniform="true" reflectance="0.2"/>
    <material name="servo_body" rgba="0.16 0.17 0.20 1" specular="0.25"
              shininess="0.6"/>
    <material name="servo_face" rgba="0.28 0.30 0.34 1" specular="0.30"
              shininess="0.8"/>
    <material name="servo_hub" rgba="0.72 0.74 0.78 1" specular="0.45"
              shininess="0.9"/>
  </asset>

  <worldbody>
    <light name="sun" pos="0 0 4.5" dir="0 0 -1" directional="true"
           diffuse="0.9 0.9 0.9" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.45"
          material="grid" condim="3"/>
    <!-- Visual tip target for the +90° elbow pose: shown on the viewer-left side. -->
    <site name="target" type="sphere" pos="-1.0 0 0.7" size="0.07"
          rgba="0.25 0.9 0.35 0.45"/>
    <camera name="fixed" pos="0 -5 1.8" zaxis="0 -1 0"/>
    <camera name="lookat" mode="targetbodycom" target="upper_arm" pos="0 -2.2 2.4"/>

    <body name="upper_arm" pos="0 0 1.7">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.35"/>
      <geom name="upper_arm_decoration" material="servo_face" type="cylinder"
            fromto="0 -0.06 0 0 0.06 0" size="0.055" mass="0.0"/>
      <geom name="upper_arm" type="capsule"
            fromto="0 0 0 0 0 -1"
            size="0.05"
            material="body"
            mass="{upper_mass}"/>

      <body name="lower_arm" pos="0 0 -1">
        <joint name="elbow" type="hinge" axis="0 1 0"
               range="-175 175" damping="0.10"/>
        <geom name="elbow_hub" type="cylinder"
              fromto="0 -0.05 0 0 0.05 0"
              size="0.052" material="servo_hub" mass="0.0"/>
        <geom name="lower_arm" type="capsule"
              fromto="0 0 0 0 0 -1"
              size="0.049"
              material="body"
              mass="{lower_mass}"/>
        <geom name="tip_payload" type="sphere"
              pos="0 0 -1"
              size="0.070"
              material="servo_hub"
              mass="{payload_mass}"/>
        <site name="tip" pos="0 0 -1" size="0.018" rgba="0.95 0.3 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="a0" joint="elbow" gear="1"/>
  </actuator>
  <custom>
    <text name="motor_a0"     data="motor_spec:faulhaber_2264w024bp4"/>
    <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
  </custom>
  <sensor>
    <user name="motor_current"  dim="1" cutoff="120.0"/>
    <user name="bus_voltage"    dim="1" cutoff="30.0"/>
    <user name="elec_power_w"   dim="1" cutoff="3000.0"/>
    <user name="battery_soc"    dim="1" cutoff="1.0"/>
    <user name="winding_temp_c" dim="1" cutoff="125.0"/>
  </sensor>
</mujoco>
"""

# Sensor names matching _XML_TEMPLATE order — used to find sensordata slots.
_SENSOR_NAMES = [
    "motor_current",
    "bus_voltage",
    "elec_power_w",
    "battery_soc",
    "winding_temp_c",
]


class Scenario(NamedTuple):
    label: str
    payload_mass: float
    hold_torque: float


def _hold_torque_for_payload(payload_mass: float) -> float:
    return _GRAVITY * (_LOWER_ARM_MASS * (_LINK_LENGTH / 2.0) + payload_mass * _LINK_LENGTH)


_LIGHT = Scenario("Light payload (25 g)", 0.025,
                  _hold_torque_for_payload(0.025))
_HEAVY = Scenario("Heavy payload (130 g)", 0.13,
                  _hold_torque_for_payload(0.13))


def gravity_feedforward_torque(
    scenario: Scenario,
    shoulder_q: float,
    elbow_q: float,
) -> float:
    """Return the gravity-compensation torque for the current Acrobot pose."""
    lower_abs_angle = shoulder_q + elbow_q
    effective_lever = _LOWER_ARM_MASS * (_LINK_LENGTH / 2.0) + scenario.payload_mass * _LINK_LENGTH
    return _GRAVITY * effective_lever * math.sin(lower_abs_angle)

# ---------------------------------------------------------------------------
# HUD text helpers (only used when Handle.set_texts() is available)
# ---------------------------------------------------------------------------
_LABEL_FONT  = mujoco.mjtFontScale.mjFONTSCALE_150
_STATUS_FONT = mujoco.mjtFontScale.mjFONTSCALE_150
_TOPRIGHT    = mujoco.mjtGridPos.mjGRID_TOPRIGHT
_TOPLEFT     = mujoco.mjtGridPos.mjGRID_TOPLEFT


def _build_hud(scenario: Scenario, state, step: int, n_steps: int) -> list:
    """Return (font, gridpos, labels, values) tuples for Handle.set_texts()."""
    torque   = float(state.torques[0])
    current  = float(state.currents[0])
    voltage  = float(state.bus_voltage)
    soc      = float(state.soc)
    temp     = float(state.temperatures[0])
    shoulder = math.degrees(float(state.qpos[0]))
    elbow    = math.degrees(float(state.qpos[1]))
    elbow_err = math.degrees(_TARGET_ELBOW - float(state.qpos[1]))
    power    = abs(current * voltage)

    stall = abs(torque) >= _PEAK_TORQUE * 0.99
    hot   = temp >= _MAX_TEMP * 0.80

    labels = (
        "Motor"
        "\nBattery"
        "\n "
        "\nTime"
        "\nShoulder"
        "\nElbow"
        "\nElbow err"
        "\nTorque"
        "\nCurrent"
        "\nVoltage"
        "\nPower"
        "\nSOC"
        "\nWinding temp"
    )
    values = (
        "Faulhaber 2264W024BP4"
        "\nUnitree G1 9 Ah"
        "\n "
        f"\n{state.time:>8.3f} s"
        f"\n{shoulder:>+8.2f} deg"
        f"\n{elbow:>+8.2f} deg"
        f"\n{elbow_err:>+8.2f} deg"
        f"\n{torque:>+8.4f} Nm{'  !! STALL' if stall else ''}"
        f"\n{current:>+8.2f} A"
        f"\n{voltage:>8.3f} V"
        f"\n{power:>8.1f} W"
        f"\n{soc:>8.5f}"
        f"\n{temp:>8.2f} degC{'  !! HOT' if hot else ''}"
    )

    grav_pct = scenario.hold_torque / _PEAK_TORQUE * 100.0
    status   = (
        "STALL — elbow overloaded" if scenario.hold_torque > _PEAK_TORQUE
        else "Motor holds elbow target"
    )
    banner_label = "Scenario\nElbow load\nStatus"
    banner_value = (
        f"{scenario.label}"
        f"\n{scenario.hold_torque:.3f} Nm  ({grav_pct:.0f}% of peak)"
        f"\n{status}"
    )

    return [
        (_LABEL_FONT,  _TOPRIGHT, labels,        values),
        (_STATUS_FONT, _TOPLEFT,  banner_label,  banner_value),
    ]


# ---------------------------------------------------------------------------
# Write electrical metrics into mjData.sensordata (user sensor slots)
# ---------------------------------------------------------------------------
def _write_sensor_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state,
) -> None:
    """Push electrical metrics into user-sensor slots after each mj_step.

    User sensors are not overwritten by MuJoCo during mj_step, so values
    written here persist until the next call and are visible in the Sensors
    panel when handle.sync() is called.
    """
    current = float(state.currents[0])
    voltage = float(state.bus_voltage)
    power   = abs(current * voltage)
    soc     = float(state.soc)
    temp    = float(state.temperatures[0])

    for name, value in zip(_SENSOR_NAMES, [current, voltage, power, soc, temp]):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid >= 0:
            data.sensordata[model.sensor_adr[sid]] = value


def _set_builtin_overlay_text(handle, scenario: Scenario, state, step: int, n_steps: int) -> None:
    """Show a visible bottom-left text block in the standard viewer.

    The public `set_texts()` API is only available in the repo's custom
    MuJoCo build. In the stock 3.2.x binary release we fall back to the
    built-in `load_error` overlay, which is always rendered at bottom-left.
    """
    sim = handle._get_sim()
    if sim is None:
        return

    torque   = float(state.torques[0])
    current  = float(state.currents[0])
    voltage  = float(state.bus_voltage)
    power    = abs(current * voltage)
    soc      = float(state.soc)
    temp     = float(state.temperatures[0])
    shoulder = math.degrees(float(state.qpos[0]))
    elbow    = math.degrees(float(state.qpos[1]))
    elbow_err = math.degrees(_TARGET_ELBOW - float(state.qpos[1]))

    stall = "  STALL" if abs(torque) >= _PEAK_TORQUE * 0.99 else ""
    hot   = "  HOT" if temp >= _MAX_TEMP * 0.80 else ""

    sim.load_error = (
        "Electrical acrobot\n"
        f"{scenario.label}\n"
        f"sh   {shoulder:+6.2f} deg\n"
        f"el   {elbow:+6.2f} deg\n"
        f"err  {elbow_err:+6.2f} deg\n"
        f"tau  {torque:+6.3f} Nm{stall}\n"
        f"I    {current:+6.1f} A\n"
        f"V    {voltage:6.2f} V\n"
        f"P    {power:6.0f} W\n"
        f"SOC  {soc:6.4f}\n"
        f"T    {temp:6.1f} degC{hot}"
    )


# ---------------------------------------------------------------------------
# Single-scenario visualizer run
# ---------------------------------------------------------------------------
def run_scenario_gui(
    scenario: Scenario,
    n_steps: int,
    next_requested: list[bool],
) -> None:
    """Run *scenario* in the live MuJoCo viewer for up to *n_steps* steps.

    Electrical metrics are written to user-sensor sensordata slots after each
    physics step so the built-in **Sensors** panel (press **F4**) displays live
    bars for current, voltage, power, SOC and winding temperature.
    """
    xml = _XML_TEMPLATE.format(
        upper_mass=_UPPER_ARM_MASS,
        lower_mass=_LOWER_ARM_MASS,
        payload_mass=scenario.payload_mass,
    )
    sim = SingleEnvSimulation.from_xml(xml, kp=220.0, kd=18.0)
    pos_target = np.array([_TARGET_ELBOW])

    dt       = float(sim.model.opt.timestep)
    grav_pct = scenario.hold_torque / _PEAK_TORQUE * 100.0
    stall_label = (
        "STALL — elbow motor overloaded" if scenario.hold_torque > _PEAK_TORQUE
        else "motor holds"
    )

    print()
    print("=" * 72)
    print(f"  Scenario: {scenario.label}")
    print(f"  Elbow load at 90° bend: {scenario.hold_torque:.3f} Nm "
          f"({grav_pct:.0f}% of peak {_PEAK_TORQUE} Nm)  ->  {stall_label}")
    print(f"  Steps: {n_steps}  |  dt: {dt}s  |  duration: {n_steps * dt:.1f}s")
    if _HAS_SET_TEXTS:
        print("  HUD overlay: top-right = metrics, top-left = scenario status")
    else:
        print("  Metrics: bottom-left text overlay + Sensors panel (F4) + terminal table")
    print("  Key N: skip to next scenario   Esc / close: exit")
    print("=" * 72)

    # Terminal column header
    print(f"  {'time(s)':>7}  {'sh(d)':>7}  {'el_err(d)':>10}  {'torq(Nm)':>9}  "
          f"{'I(A)':>7}  {'V(V)':>7}  {'P(W)':>7}  {'SOC':>6}  {'T(°C)':>7}")
    print("  " + "-" * 74)

    next_requested[0] = False

    def key_callback(keycode: int) -> None:
        # GLFW key codes: N = 78
        if keycode == 78:
            next_requested[0] = True

    state = None
    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=key_callback,
    ) as handle:
        start_wall = time.time()
        sim_clock  = 0.0
        report_every = max(1, n_steps // 20)

        for step in range(n_steps):
            if not handle.is_running() or next_requested[0]:
                break

            # ── advance physics and write sensor data ────────────────────
            with handle.lock():
                gravity_ff = gravity_feedforward_torque(
                    scenario,
                    float(sim.data.qpos[0]),
                    float(sim.data.qpos[1]),
                )
                state = sim.step(
                    pos_targets=pos_target,
                    effort_targets=np.array([gravity_ff]),
                )
                # Write metrics AFTER mj_step so they are not overwritten.
                _write_sensor_data(sim.model, sim.data, state)

            # ── optional text overlay ────────────────────────────────────
            if _HAS_SET_TEXTS:
                handle.set_texts(_build_hud(scenario, state, step, n_steps))
            else:
                _set_builtin_overlay_text(handle, scenario, state, step, n_steps)

            # Sync copies all of mjData (including sensordata) to renderer.
            handle.sync()

            # ── terminal table ───────────────────────────────────────────
            if step % report_every == 0 or step == n_steps - 1:
                torque  = float(state.torques[0])
                current = float(state.currents[0])
                voltage = float(state.bus_voltage)
                power   = abs(current * voltage)
                shoulder = math.degrees(float(state.qpos[0]))
                elbow_err = math.degrees(_TARGET_ELBOW - float(state.qpos[1]))
                stall_f = " STALL" if abs(torque) >= _PEAK_TORQUE * 0.99 else ""
                hot_f   = " HOT"   if state.temperatures[0] >= _MAX_TEMP * 0.80 else ""
                print(f"  {state.time:>7.3f}  {shoulder:>+7.2f}  {elbow_err:>+10.2f}  "
                      f"{torque:>+9.4f}  {current:>+7.2f}  "
                      f"{voltage:>7.3f}  {power:>7.1f}  "
                      f"{state.soc:>6.4f}  {state.temperatures[0]:>7.2f}"
                      f"{stall_f}{hot_f}")

            # ── pace to real time ────────────────────────────────────────
            sim_clock += dt
            lag = sim_clock - (time.time() - start_wall)
            if lag > 0:
                time.sleep(lag)

    if state is None:
        return

    # ── final summary ──────────────────────────────────────────────────────
    torque  = float(state.torques[0])
    temp    = float(state.temperatures[0])
    soc     = float(state.soc)
    voltage = float(state.bus_voltage)
    current = float(state.currents[0])
    elbow_err = math.degrees(_TARGET_ELBOW - float(state.qpos[1]))
    print()
    print(f"  Summary:  elbow_err={elbow_err:+.2f} deg  "
          f"I={current:+.1f} A  "
          f"V={voltage:.3f} V  "
          f"SOC={soc:.5f}  "
          f"Temp={temp:.1f} degC")
    if abs(torque) >= _PEAK_TORQUE * 0.99:
        print("  * Motor stalled: torque clamped near peak.")
        print(f"    Winding heated to {temp:.1f} degC "
              f"({temp / _MAX_TEMP * 100:.1f}% of {_MAX_TEMP:.0f} degC limit).")
    else:
        print(f"  * Motor held the elbow target: final error = {abs(elbow_err):.2f} deg.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Electrical motor GUI demo: Acrobot elbow with live metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--light", action="store_true",
                           help="Light arm scenario only.")
    arm_group.add_argument("--heavy", action="store_true",
                           help="Heavy arm scenario only.")
    parser.add_argument("--steps", type=int, default=2500,
                        help="Physics steps per scenario (default: 2500).")
    args = parser.parse_args()

    dt = 0.002  # matches XML template timestep
    total_time = args.steps * dt
    hud_status = (
        "HUD overlay enabled" if _HAS_SET_TEXTS
        else "bottom-left overlay + sensors panel (F4) + terminal"
    )

    print(f"\nElectrical Motor GUI Demo")
    print(f"  Motor:   Faulhaber 2264W024BP4  (peak torque = {_PEAK_TORQUE} Nm)")
    print(f"  Battery: Unitree G1 9 Ah  (24 V nominal)")
    print(f"  Steps:   {args.steps} x {dt}s = {total_time:.1f}s per scenario")
    print("  Model:   Acrobot-style two-link scene based on dm_control/suite/acrobot.xml")
    print(f"  Target:  elbow = {math.degrees(_TARGET_ELBOW):.0f}° bend; shoulder is passive")
    print(f"  Metrics: {hud_status}")
    print()
    print("  Viewer controls:")
    print("    F4     toggle Sensors panel (current, voltage, power, SOC, temp)")
    print("    N      skip to next scenario")
    print("    Esc    exit")

    if args.light:
        scenarios = [_LIGHT]
    elif args.heavy:
        scenarios = [_HEAVY]
    else:
        scenarios = [_LIGHT, _HEAVY]

    next_requested: list[bool] = [False]

    for scenario in scenarios:
        run_scenario_gui(scenario, args.steps, next_requested)
        if len(scenarios) > 1:
            print("\n  [Next scenario opens in 2 s...  close the viewer window first]")
            time.sleep(2.0)

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
