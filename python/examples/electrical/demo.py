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
"""Phase 7 demonstration: Acrobot elbow driven by a Faulhaber 2264W motor.

This version replaces the single rigid arm with an **Acrobot-style two-link
mechanism** based on the MuJoCo model in
``dm_control/suite/acrobot.xml``. The shoulder joint is passive and the elbow
joint is driven through ``mujoco.electrical``.

Both scenarios start from the natural hanging pose and the controller asks the
elbow motor to hold a **90° bend** so the lower link sticks out roughly
horizontal. A simple gravity-compensation feed-forward term is added each step
using the current lower-link angle in world coordinates.

Two tip-payload cases illustrate motor limits:

  Light payload  (25 g at the tip)  → elbow load ≈ 0.74 N·m  (56 % of peak)
                   Motor holds the bend with clearly lower steady current.

  Heavy payload (100 g at the tip)  → elbow load ≈ 1.47 N·m (112 % of peak)
                   Motor saturates at peak torque, the elbow droops, and the
                   winding heats up.

Usage
-----
    python demo.py                  # headless table (both scenarios)
    python demo.py --light          # light payload only
    python demo.py --heavy          # heavy payload only
    python demo.py --steps 2500     # change number of steps (default 2500)
    python demo.py --plot           # matplotlib plots (requires matplotlib)
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Check for MuJoCo C extension early with a clear message.
# ---------------------------------------------------------------------------
try:
    import mujoco  # noqa: F401
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

from mujoco.electrical import SingleEnvSimulation  # noqa: E402

# ---------------------------------------------------------------------------
# Acrobot-style XML template — inspired by dm_control/suite/acrobot.xml.
# Only the elbow is actuated; the shoulder remains passive.
# ---------------------------------------------------------------------------
_XML_TEMPLATE = """\
<mujoco model="electrical_acrobot_demo">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <worldbody>
    <light name="light" pos="0 0 6"/>
    <geom name="floor" type="plane" size="3 3 0.2" pos="0 0 -0.45"/>

    <body name="upper_arm" pos="0 0 1.7">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.35"/>
      <geom name="upper_arm_decoration" type="cylinder"
            fromto="0 -0.06 0 0 0.06 0"
            size="0.051" mass="0.0"/>
      <geom name="upper_arm" type="capsule"
            fromto="0 0 0 0 0 -1"
            size="0.05"
            mass="{upper_mass}"/>

      <body name="lower_arm" pos="0 0 -1">
        <joint name="elbow" type="hinge" axis="0 1 0"
               range="-175 175" damping="0.10"/>
        <geom name="lower_arm" type="capsule"
              fromto="0 0 0 0 0 -1"
              size="0.049"
              mass="{lower_mass}"/>
        <geom name="tip_payload" type="sphere"
              pos="0 0 -1"
              size="0.070"
              mass="{payload_mass}"/>
        <site name="tip" pos="0 0 -1" size="0.01"/>
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
</mujoco>
"""

# ---------------------------------------------------------------------------
# Physics reference values (for annotation in output)
# ---------------------------------------------------------------------------
_LINK_LENGTH      = 1.0           # m  (matches dm_control acrobot link lengths)
_UPPER_ARM_MASS   = 0.12          # kg
_LOWER_ARM_MASS   = 0.10          # kg
_GRAVITY          = 9.81          # m/s²
_TARGET_ELBOW     = math.pi / 2   # rad → hold a 90° bend at the elbow
_PEAK_TORQUE      = 1.311         # N·m  Faulhaber 2264W
_MAX_TEMP         = 125.0         # °C   Faulhaber winding limit


class Scenario(NamedTuple):
    label: str
    payload_mass: float       # kg
    hold_torque: float        # N·m


def _hold_torque_for_payload(payload_mass: float) -> float:
    """Approximate elbow torque needed at a 90° bent pose."""
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
    """Return the elbow gravity-compensation torque for the current pose.

    The Acrobot lower-link load depends on the lower arm's absolute angle in
    world coordinates, ``shoulder_q + elbow_q``. The elbow must support the
    lower-link COM and the tip payload.
    """
    lower_abs_angle = shoulder_q + elbow_q
    effective_lever = _LOWER_ARM_MASS * (_LINK_LENGTH / 2.0) + scenario.payload_mass * _LINK_LENGTH
    return _GRAVITY * effective_lever * math.sin(lower_abs_angle)


# ---------------------------------------------------------------------------
# Run one scenario — returns history for optional plotting.
# ---------------------------------------------------------------------------
def run_scenario(scenario: Scenario, n_steps: int, verbose: bool = True):
    """Run *n_steps* with *scenario* and return a dict of time-series."""
    xml = _XML_TEMPLATE.format(
        upper_mass=_UPPER_ARM_MASS,
        lower_mass=_LOWER_ARM_MASS,
        payload_mass=scenario.payload_mass,
    )

    sim = SingleEnvSimulation.from_xml(xml, kp=220.0, kd=18.0)
    pos_target = np.array([_TARGET_ELBOW])

    hist: dict[str, list] = {
        "time": [],
        "shoulder_q": [],
        "elbow_q": [],
        "elbow_err_deg": [],
        "torque": [],
        "current": [],
        "soc": [],
        "bus_v": [],
        "temp": [],
    }

    grav_pct = scenario.hold_torque / _PEAK_TORQUE * 100.0
    stall_label = (
        "STALL — elbow motor overloaded"
        if scenario.hold_torque > _PEAK_TORQUE else "motor holds"
    )

    if verbose:
        print()
        print("=" * 72)
        print(f"  {scenario.label}")
        print(f"  Elbow load at 90° bend: {scenario.hold_torque:.3f} N·m  "
              f"({grav_pct:.0f}% of peak {_PEAK_TORQUE} N·m)  →  {stall_label}")
        print("=" * 72)
        print(f"  {'step':>5}  {'time(s)':>7}  {'shoulder(°)':>11}  {'elbow_err(°)':>12}  "
              f"{'torque(Nm)':>10}  {'I(A)':>7}  {'SOC':>6}  {'V_bus(V)':>8}  {'T_winding(°C)':>13}")
        print("  " + "-" * 96)

    for step in range(n_steps):
        gravity_ff = gravity_feedforward_torque(
            scenario,
            float(sim.data.qpos[0]),
            float(sim.data.qpos[1]),
        )
        state = sim.step(
            pos_targets=pos_target,
            effort_targets=np.array([gravity_ff]),
        )

        shoulder_deg = math.degrees(float(state.qpos[0]))
        elbow_deg = math.degrees(float(state.qpos[1]))
        elbow_err_deg = math.degrees(_TARGET_ELBOW - float(state.qpos[1]))

        hist["time"].append(state.time)
        hist["shoulder_q"].append(float(state.qpos[0]))
        hist["elbow_q"].append(float(state.qpos[1]))
        hist["elbow_err_deg"].append(elbow_err_deg)
        hist["torque"].append(float(state.torques[0]))
        hist["current"].append(float(state.currents[0]))
        hist["soc"].append(state.soc)
        hist["bus_v"].append(state.bus_voltage)
        hist["temp"].append(float(state.temperatures[0]))

        report_interval = max(1, n_steps // 20)
        if verbose and (step % report_interval == 0 or step == n_steps - 1):
            temp_flag = " ⚑ HOT" if state.temperatures[0] > _MAX_TEMP * 0.9 else ""
            stall_flag = " ✗ stall" if abs(state.torques[0]) >= _PEAK_TORQUE * 0.99 else ""
            print(f"  {step:>5}  {state.time:>7.3f}  {shoulder_deg:>+11.2f}  {elbow_err_deg:>+12.2f}  "
                  f"{state.torques[0]:>+10.4f}  {state.currents[0]:>+7.2f}  {state.soc:>6.4f}  "
                  f"{state.bus_voltage:>8.3f}  {state.temperatures[0]:>13.2f}"
                  f"{stall_flag}{temp_flag}")

    if verbose:
        final = hist
        avg_abs_err = np.mean(np.abs(final["elbow_err_deg"]))
        max_temp = max(final["temp"])
        print()
        print(f"  Summary: avg|elbow_error|={avg_abs_err:.2f}°  "
              f"peak_temp={max_temp:.1f}°C  "
              f"final_SOC={final['soc'][-1]:.5f}  "
              f"final_V={final['bus_v'][-1]:.3f}V")
        if scenario.hold_torque > _PEAK_TORQUE:
            print("  ★  Motor stalled as expected: torque clamped near peak.")
            print(f"     Winding heated to {max_temp:.1f}°C "
                  f"({max_temp / _MAX_TEMP * 100:.1f}% of {_MAX_TEMP:.0f}°C limit).")
        else:
            final_err = abs(final["elbow_err_deg"][-1])
            print(f"  ★  Motor held the elbow target: final error = {final_err:.2f}°.")

    return hist


# ---------------------------------------------------------------------------
# Optional matplotlib visualisation.
# ---------------------------------------------------------------------------
def plot_histories(histories: dict[str, dict], n_steps: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plots. "
              "Install with: pip install matplotlib", file=sys.stderr)
        return

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(
        "Electrical Motor Demo — Faulhaber 2264W + Unitree G1 Battery\n"
        "Acrobot elbow target = 90° bend (dm_control-inspired model)",
        fontsize=11,
    )

    colors = {"Light payload (25 g)": "#2196F3", "Heavy payload (100 g)": "#F44336"}

    for label, hist in histories.items():
        t = hist["time"]
        col = colors.get(label, "gray")
        axes[0].plot(t, np.degrees(hist["elbow_q"]),   color=col, label=label)
        axes[1].plot(t, np.degrees(hist["shoulder_q"]), color=col, label=label)
        axes[2].plot(t, hist["torque"],                 color=col, label=label)
        axes[3].plot(t, hist["current"],                color=col, label=label)
        axes[4].plot(t, hist["temp"],                   color=col, label=label)

    # Annotations
    axes[0].axhline(math.degrees(_TARGET_ELBOW), color="k", lw=0.8, ls="--", label="target elbow")
    axes[2].axhline(_PEAK_TORQUE, color="orange", lw=1.2, ls="--",
                    label=f"peak torque ({_PEAK_TORQUE} N·m)")
    axes[2].axhline(-_PEAK_TORQUE, color="orange", lw=1.2, ls="--")
    axes[4].axhline(_MAX_TEMP, color="red", lw=1.2, ls="--",
                    label=f"winding limit ({_MAX_TEMP}°C)")

    labels = [
        ("Elbow angle (°)",         "Elbow angle"),
        ("Shoulder angle (°)",      "Shoulder angle"),
        ("Torque (N·m)",            "Torque"),
        ("Winding current (A)",     "Current"),
        ("Winding temperature (°C)","Temperature"),
    ]
    for ax, (ylabel, title) in zip(axes, labels):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Electrical motor demo: Acrobot elbow with light vs heavy payload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--light", action="store_true",
                           help="Run light arm scenario only.")
    arm_group.add_argument("--heavy", action="store_true",
                           help="Run heavy arm scenario only.")
    parser.add_argument("--steps", type=int, default=2500,
                        help="Number of simulation steps (default: 300).")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib plots after simulation.")
    args = parser.parse_args()

    dt = 0.002
    total_time = args.steps * dt
    print(f"\nElectrical Motor Demo  —  {args.steps} steps × {dt}s = {total_time:.3f}s")
    print(f"Motor:   Faulhaber 2264W024BP4  (peak torque = {_PEAK_TORQUE} N·m)")
    print(f"Battery: Unitree G1 9 Ah  (24 V nominal)")
    print("Model:   Acrobot-style two-link model based on dm_control/suite/acrobot.xml")
    print(f"Target:  elbow = {math.degrees(_TARGET_ELBOW):.0f}° bend; shoulder is passive")

    scenarios = []
    if args.light:
        scenarios = [_LIGHT]
    elif args.heavy:
        scenarios = [_HEAVY]
    else:
        scenarios = [_LIGHT, _HEAVY]

    histories = {}
    for scenario in scenarios:
        hist = run_scenario(scenario, args.steps, verbose=True)
        histories[scenario.label] = hist

    if args.plot:
        plot_histories(histories, args.steps)


if __name__ == "__main__":
    main()
