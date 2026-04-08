.. _ElectricalMotorsSim:

*************************************
Electrical Motor & Battery Simulation
*************************************

MuJoCo ships with a pure Python/NumPy sub-package,
``mujoco.electrical``, that models DC-motor drive electronics and
lithium-based battery packs in a single-environment simulation loop.
A companion native C++ actuator plugin is provided for use cases that
require the low-level MuJoCo plugin API.

.. _ElectricalOverview:

Overview
========

The module is located at ``python/mujoco/electrical/`` inside the
repository and is organised into four layers:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Layer
     - Responsibility
   * - **Data**
     - Typed dataclasses (``MotorSpecification``, ``BatterySpecification``)
       that map directly to the community JSON asset format.
   * - **Database**
     - ``MotorDatabase`` / ``BatteryDatabase`` — find and load specs from
       local paths, environment variables, or remote community repos with
       MD5-verified caching.
   * - **XML integration**
     - Parse and write ``motor_spec:`` / ``battery_spec:`` tags inside a
       ``<custom>`` block of a MuJoCo scene spec.
   * - **CLI**
     - ``download_specs.py`` — pre-fetch and inspect specs from the
       command line.

.. _ElectricalInstall:

Installation
============

The sub-package requires **NumPy** only (no PyTorch, no GPU):

.. code-block:: shell

   pip install numpy

No separate installation step is needed when building MuJoCo from
source; ``mujoco.electrical`` is part of the ``mujoco`` Python package.

.. _ElectricalDataLayer:

Data layer
==========

.. _ElectricalMotorSpec:

MotorSpecification
------------------

``mujoco.electrical.MotorSpecification`` is a dataclass that holds all
fields from the community motor-asset JSON schema.  Fields that may be
absent in real-world datasheets are typed ``Optional[float]``.

.. code-block:: python

   from mujoco.electrical import MotorSpecification

   spec = MotorSpecification(
       motor_id          = "faulhaber_2264w024bp4",
       manufacturer      = "Faulhaber",
       model             = "2264W024BP4",
       gear_ratio        = 1.0,
       motor_constant_kt = 0.0118,   # N·m/A
       motor_constant_ke = 0.0118,   # V·s/rad
       peak_torque       = 1.311,    # N·m
       stall_torque      = 1.311,
       continuous_torque = 0.059,
       no_load_speed     = 2209.6,   # rad/s
       no_load_current   = 0.261,    # A
       voltage_range     = [0.0, 24.0],
       resistance        = 0.22,     # Ω
       inductance        = 2.4e-5,   # H
       thermal_resistance    = 5.0,   # °C/W
       thermal_time_constant = 950.0, # s
       max_winding_temperature = 125.0,
   )

   print(spec.has_rl_circuit)  # True
   print(spec.has_thermal)     # True

Two boolean helper properties indicate which physics paths are
available at runtime:

``has_rl_circuit``
   ``True`` when **both** ``resistance`` and ``inductance`` are
   non-``None``.  When ``False`` the simulation uses the *degraded
   path* (direct torque clamping, approximate current).

``has_thermal``
   ``True`` when **both** ``thermal_resistance`` and
   ``thermal_time_constant`` are non-``None``.

The ``from_dict()`` class method constructs a spec from a
JSON-parsed dictionary; unknown keys (future asset extensions) are
silently ignored:

.. code-block:: python

   import json
   with open("my_motor.json") as f:
       spec = MotorSpecification.from_dict(json.load(f))

.. _ElectricalMotorSpecFields:

Required fields
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Unit
     - Description
   * - ``motor_id``
     - —
     - Unique identifier matching the JSON filename.
   * - ``manufacturer``
     - —
     - Manufacturer name.
   * - ``model``
     - —
     - Part / model number.
   * - ``gear_ratio``
     - —
     - Output-to-motor ratio (1.0 = direct drive).
   * - ``motor_constant_kt``
     - N·m/A
     - Torque constant :math:`K_t`.
   * - ``motor_constant_ke``
     - V·s/rad
     - Back-EMF constant :math:`K_e`.
   * - ``peak_torque``
     - N·m
     - Peak instantaneous output torque.
   * - ``stall_torque``
     - N·m
     - Stall torque.
   * - ``continuous_torque``
     - N·m
     - Continuous rated torque.
   * - ``no_load_speed``
     - rad/s
     - No-load angular speed.
   * - ``no_load_current``
     - A
     - No-load current draw.
   * - ``voltage_range``
     - V
     - ``[v_min, v_max]`` operating voltage range.

Optional RL-circuit fields
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Unit
     - Description
   * - ``resistance``
     - Ω
     - Winding resistance :math:`R`.  ``None`` → degraded path.
   * - ``inductance``
     - H
     - Winding inductance :math:`L`.  ``None`` → degraded path.

Optional thermal fields
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Unit
     - Description
   * - ``thermal_resistance``
     - °C/W
     - Winding-to-ambient thermal resistance :math:`R_{th}`.
   * - ``thermal_time_constant``
     - s
     - Thermal time constant :math:`\tau_{th}`.
   * - ``max_winding_temperature``
     - °C
     - Maximum allowable winding temperature.

.. _ElectricalBatterySpec:

BatterySpecification
--------------------

``mujoco.electrical.BatterySpecification`` covers cell chemistry,
pack configuration, discharge curves, and thermal properties.  Pack
voltages (``nominal_voltage``, ``max_voltage``, ``min_voltage``) and
energy capacity (``energy_wh``) are derived automatically in
``__post_init__``.

.. code-block:: python

   from mujoco.electrical import BatterySpecification

   spec = BatterySpecification(
       battery_id        = "unitree_g1_9ah",
       manufacturer      = "Unitree Robotics",
       model             = "G1 High-Performance Li-ion Battery",
       chemistry         = "Li-ion",
       cells_series      = 6,
       cells_parallel    = 1,
       nominal_cell_voltage = 3.6,
       max_cell_voltage     = 4.2,
       min_cell_voltage     = 2.5,
       capacity_ah          = 9.0,
       internal_resistance  = 0.015,   # Ω
       max_continuous_current = 15.12, # A
   )

   print(spec.nominal_voltage)   # 21.6 V  (3.6 × 6)
   print(spec.energy_wh)         # 194.4 Wh

Default curves are supplied when ``ocv_curve`` or
``internal_resistance_soc_curve`` are absent from the JSON asset.
The ``ocv_soc``, ``ocv_v``, ``r_soc``, and ``r_mult`` properties
expose flat lists suitable for ``numpy.interp``.

.. _ElectricalDatabase:

Database
========

Both ``MotorDatabase`` and ``BatteryDatabase`` resolve specs using
the following priority order (highest to lowest):

1. **Explicit path** — ``db.load(id, path="/path/to/my.json")``
2. **Environment variable** — ``MUJOCO_MOTOR_PATH`` or
   ``MUJOCO_BATTERY_PATH`` (colon-separated directory list).
3. **User directory** — ``~/.mujoco/motors/`` or
   ``~/.mujoco/batteries/``.
4. **Remote community repository** — downloaded over HTTPS and cached
   to ``~/.mujoco/cache/`` (served from cache on subsequent calls).

Community repository URLs
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Base URL
   * - Motors
     - ``https://raw.githubusercontent.com/robomotic/mujoco-motors/master/motor_assets``
   * - Batteries
     - ``https://raw.githubusercontent.com/robomotic/mujoco-batteries/master/battery_assets``

The vendor prefix is inferred from the first underscore-delimited
token of the spec ID.  For example:

* ``faulhaber_2264w024bp4`` → ``motor_assets/faulhaber/faulhaber_2264w024bp4.json``
* ``maxon_ec_i_40_488607``  → ``motor_assets/maxon/maxon_ec_i_40_488607.json``
* ``unitree_g1_9ah``        → ``battery_assets/unitree/unitree_g1_9ah.json``

If vendor inference fails, ``MotorDatabase`` falls back to scanning
the GitHub repository tree API before raising ``FileNotFoundError``.

Usage
-----

.. code-block:: python

   from mujoco.electrical import MotorDatabase, BatteryDatabase

   motor_db   = MotorDatabase()
   battery_db = BatteryDatabase()

   # Load by ID — fetches from remote on first call, cached thereafter
   faulhaber = motor_db.load("faulhaber_2264w024bp4")
   maxon     = motor_db.load("maxon_ec_i_40_488607")
   battery   = battery_db.load("unitree_g1_9ah")

   # Force a fresh download (bypasses local cache)
   battery   = battery_db.load("unitree_g1_9ah", force=True)

   # Load from an explicit local file
   custom = motor_db.load("my_motor", path="/path/to/my_motor.json")

   # Search locally-cached specs
   results = motor_db.search("maxon")

   # List all cached spec IDs
   print(motor_db.list_cached())

Caching behaviour
-----------------

Downloaded JSON files are stored in ``~/.mujoco/cache/`` with a
filename derived from the MD5 of the source URL.  A companion
``<hash>.md5`` sidecar records the content MD5 for integrity
checking.  The cache is trusted on all subsequent loads; pass
``force=True`` to re-download unconditionally.

Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Effect
   * - ``MUJOCO_MOTOR_PATH``
     - Colon-separated list of directories searched before the remote
       repo for motor specs.
   * - ``MUJOCO_BATTERY_PATH``
     - Colon-separated list of directories searched before the remote
       repo for battery specs.

.. _ElectricalXML:

XML auto-discovery
==================

Motor and battery specs are referenced inside the ``<custom>`` block
of a MuJoCo XML model using ``<text>`` elements with the following
naming convention:

.. code-block:: xml

   <mujoco>
     <custom>
       <!-- Motor: name = "motor_{actuator_name}"  data = "motor_spec:{motor_id}" -->
       <text name="motor_a0"     data="motor_spec:faulhaber_2264w024bp4"/>

       <!-- Battery: name = "battery_{label}"  data = "battery_spec:{battery_id}" -->
       <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
     </custom>
   </mujoco>

The helper functions in ``mujoco.electrical.xml_integration`` read
and write these tags from a :class:`mujoco.MjSpec` object:

.. code-block:: python

   import mujoco
   from mujoco.electrical import (
       parse_motor_specs_from_xml,
       parse_battery_specs_from_xml,
       write_motor_spec_to_xml,
       write_battery_spec_to_xml,
   )

   spec = mujoco.MjSpec.from_string(xml_string)

   # Parse existing tags → {actuator_name: motor_id}
   motor_map   = parse_motor_specs_from_xml(spec)
   battery_map = parse_battery_specs_from_xml(spec)

   # Add new tags programmatically
   write_motor_spec_to_xml(spec, actuator_name="a1", motor_id="maxon_ec_i_40_488607")
   write_battery_spec_to_xml(spec, label="aux",   battery_id="unitree_g1_9ah")

``SingleEnvSimulation.from_xml()`` uses these helpers to auto-wire motor and battery objects from a
model string without any Python-side configuration.

C++ Actuator Plugin
===================

As an alternative to the Python simulation, MuJoCo provides a native C++ plugin implementation of the electrical motor model: ``mujoco.actuator.electrical_motor``.
This allows the electrical motor simulation to run entirely within the core MuJoCo C/C++ stepping loop without any Python callbacks, which is critical for reinforcement learning and performance-sensitive C++ applications.

To use the plugin, add the `<extension>` and configure a `<plugin>` element inside an actuator:

.. code-block:: xml

   <mujoco>
     <extension>
       <plugin plugin="mujoco.actuator.electrical_motor"/>
     </extension>
     
     <actuator>
       <plugin plugin="mujoco.actuator.electrical_motor" actdim="2">
         <config key="motor_id" value="faulhaber_2264w024bp4"/>
         <config key="kp" value="100.0"/>
         <config key="kd" value="10.0"/>
       </plugin>
     </actuator>
   </mujoco>

Plugin Attributes:
* ``motor_id`` (required) — Unique identifier to fetch the spec from the JSON database.
* ``kp`` (optional) — Proportional gain (default 100.0).
* ``kd`` (optional) — Derivative gain (default 10.0).
* ``spec_path`` (optional) — Explicit local file path to the JSON spec.

The plugin tracks current/load state variables and temperature by defining internal physics states inside the `mjData.act` array. The actuator requires `actdim="2"` to hold the previous current and the coil temperature variables.

.. _ElectricalDemo:

Demo: stall, current limit, and heating
=======================================

The repository includes two end-to-end examples for the electrical motor
package, both found in ``python/examples/electrical/``:

``demo.py``
  Headless simulation that prints a live table to the terminal and optionally
  plots results with Matplotlib (``--plot``).

``demo_visualizer.py``
  Interactive demonstration that opens the **standard MuJoCo viewer window**
  (via :py:func:`mujoco.viewer.launch_passive`) and overlays live electrical
  metrics — torque, winding current, bus voltage, battery SOC, and winding
  temperature — directly on the 3-D viewport while the simulation runs.
  Use this version to see the motor physics in real time.

Both demos now build an **Acrobot-style two-link MuJoCo model** based on
``dm_control/suite/acrobot.xml``. The shoulder joint is passive, the elbow is
motorized through ``mujoco.electrical``, and the controller asks the elbow to
hold a ``90°`` bend while the mechanism hangs under gravity.

The demo compares two payload cases:

* **Light payload** (25 g at the tip): the elbow hold load is about
  ``0.736 N·m`` (56% of the motor's ``1.311 N·m`` peak), so the controller can
  hold the target bend. With gravity feed-forward enabled, the steady winding
  current settles around ``55–60 A`` and the elbow error stays near ``0°``.
* **Heavy payload** (130 g at the tip): the elbow hold load is about
  ``1.766 N·m`` (135% of peak), so the actuator saturates at peak torque, the
  elbow cannot perfectly hold the target bend, and the winding temperature
  rises steadily during stall.

This makes the example useful as a quick sanity check for three behaviors at
once: lower steady current in the supported light-load case, torque
saturation under overload (about ``114 A`` in the heavy-payload case), and
thermal accumulation during stall.

Headless demo
-------------

Run the headless version from the repository root:

.. code-block:: shell

   PYTHONPATH=python python3 python/examples/electrical/demo.py
   PYTHONPATH=python python3 python/examples/electrical/demo.py --light
   PYTHONPATH=python python3 python/examples/electrical/demo.py --heavy --steps 15000
   PYTHONPATH=python python3 python/examples/electrical/demo.py --plot

Notes:

* The default run length is ``2500`` steps at ``dt = 0.002`` s, i.e. ``5`` s.
* ``--plot`` adds a Matplotlib view of angle, torque, current, bus voltage,
  state of charge, and winding temperature.
* The scene uses an Acrobot-style body hierarchy: a passive ``shoulder`` joint
  plus an electrically driven ``elbow`` joint.
* The elbow target is a ``90°`` bend (``π/2`` rad), while the shoulder is free
  to react dynamically under gravity.

.. _ElectricalDemoVisualizer:

GUI demo with live metrics overlay
-----------------------------------

The visualizer demo opens the standard MuJoCo viewer with
:py:func:`mujoco.viewer.launch_passive` and exposes the electrical quantities
through **user sensors** written into ``mjData.sensordata``. This means the
values can be displayed by MuJoCo's built-in **Sensors** panel while the demo
is running.

Launch commands
^^^^^^^^^^^^^^^

Run the demo from the repository root:

.. code-block:: shell

   # Run both scenarios back-to-back (default: light first, then heavy)
   PYTHONPATH=python python3 python/examples/electrical/demo_visualizer.py

   # Light payload only: elbow reaches the 90° target with low steady current
   PYTHONPATH=python python3 python/examples/electrical/demo_visualizer.py --light

   # Heavy payload only: elbow motor saturates, stalls, and heats up
   PYTHONPATH=python python3 python/examples/electrical/demo_visualizer.py --heavy

   # Longer run (steps × 0.002 s each)
   PYTHONPATH=python python3 python/examples/electrical/demo_visualizer.py --heavy --steps 5000

In other words:

* **No flag** → run **both** the light and heavy payload examples.
* ``--light`` → run only the **light payload** case.
* ``--heavy`` → run only the **heavy payload** case.

The light payload is the "motor holds" example, while the heavy payload is the
"stall and heating" example.

What is displayed
^^^^^^^^^^^^^^^^^

The viewer shows a compact electrical status block directly in the viewport,
for example:

.. code-block:: text

   Electrical acrobot
   Heavy payload (130 g)
   sh   -23.00 deg
   el   +85.92 deg
   err   +4.08 deg
   tau   +1.311 Nm  STALL
   I    +114.3 A
   V      23.60 V
   P       2699 W
   SOC    0.9960
   T       28.3 degC

It also publishes five custom user-sensor channels to ``mjData.sensordata``:

* ``motor_current``
* ``bus_voltage``
* ``elec_power_w``
* ``battery_soc``
* ``winding_temp_c``

A translucent green sphere in the scene marks the ideal tip position for the
``+90°`` elbow target. It is intentionally placed on the **left** side of the
viewer so you can tell whether the Acrobot reaches the intended bend direction.

Keyboard shortcuts inside the viewer:

* **F4** — toggle the MuJoCo **Sensors** panel (reads from ``mjData.sensordata``).
* **N** — skip immediately to the next scenario.
* **Esc** / close button — exit the demo.

The physics loop in ``demo_visualizer.py`` calls
:py:class:`~mujoco.electrical.SingleEnvSimulation` exclusively. After each
step, it writes the electrical outputs into the user-sensor slots and updates
a small text overlay so the values are easy to read without opening additional
panels.

Basic math behind the demo
^^^^^^^^^^^^^^^^^^^^^^^^^^

The demo is intentionally simple enough to reason about by hand.

1. **Gravity load on the Acrobot elbow**

   When the upper arm hangs roughly downward, the elbow motor mainly supports
   the lower-link mass and the tip payload. If :math:`L` is the lower-link
   length, :math:`m_\ell` is the lower-link mass, :math:`m_p` is the tip
   payload, and :math:`\phi = q_{shoulder} + q_{elbow}` is the lower arm's
   absolute angle in world coordinates, then the elbow gravity torque is
   approximately:

   .. math::

      \tau_e \approx g\,\sin(\phi)\left(m_\ell \frac{L}{2} + m_p L\right)

   At the target bent pose, the lower arm is close to horizontal so
   :math:`\sin(\phi) \approx 1`. With :math:`L = 1.0\,\mathrm{m}` and
   :math:`m_\ell = 0.10\,\mathrm{kg}`:

   * **Light payload** (:math:`m_p = 0.025\,\mathrm{kg}`) →
     :math:`\tau_e \approx 9.81(0.10\cdot 0.5 + 0.025) \approx 0.736\,\mathrm{N\cdot m}`
   * **Heavy payload** (:math:`m_p = 0.13\,\mathrm{kg}`) →
     :math:`\tau_e \approx 9.81(0.10\cdot 0.5 + 0.13) \approx 1.766\,\mathrm{N\cdot m}`

2. **Torque-to-current relation**

   The Faulhaber motor uses:

   .. math::

      \tau = K_t I

   so the current needed for a given torque is:

   .. math::

      I \approx \frac{\tau}{K_t}

   With :math:`K_t = 0.0118\,\mathrm{N\cdot m/A}`:

   * supported light-payload torque :math:`0.736\,\mathrm{N\cdot m}` →
     :math:`I \approx 0.736 / 0.0118 \approx 62\,\mathrm{A}`
   * peak motor torque :math:`1.311\,\mathrm{N\cdot m}` →
     :math:`I \approx 1.311 / 0.0118 \approx 111\,\mathrm{A}`

   This is why the Acrobot demo shows a supported light case around
   **55–60 A**, while the overloaded heavy case climbs to roughly
   **114 A** near stall.

3. **Electrical power**

   The displayed power is the simple product:

   .. math::

      P = V I

   so a stalled heavy case at about :math:`23.6\,\mathrm{V}` and
   :math:`114\,\mathrm{A}` corresponds to about
   :math:`23.6 \times 114 \approx 2.7\,\mathrm{kW}` of electrical input.

4. **Why the heavy payload stalls**

   The motor can only deliver up to its peak torque. Since the heavy-payload
   elbow load is larger than the motor's peak capability, the controller cannot
   perfectly hold the requested 90° bend. The torque saturates, a residual
   elbow error remains, and the winding temperature rises over time.

Recording a video or GIF for sharing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For posting online, there are two easy workflows.

**Option A — use the pre-rendered files already generated in this repo**

The example exports can be found in:

.. code-block:: text

   python/examples/electrical/recordings/

When you regenerate media for the current Acrobot demo, typical outputs are:

* ``light_acrobot_demo_5s.mp4``
* ``light_acrobot_demo_5s.gif``
* ``heavy_acrobot_demo_5s.mp4``
* ``heavy_acrobot_demo_5s.gif``

**Option B — record the MuJoCo window yourself**

1. Record a short **MP4** of the MuJoCo window with your normal screen-capture
   tool (for example **OBS Studio**, **ScreenToGif**, or the Windows
   Snipping Tool screen recorder).
2. Trim the clip to the interesting portion (for example the first 5--10 s of
   the heavy-payload stall).
3. Convert the MP4 to a lightweight GIF if needed.

If you want to regenerate the four demo media files programmatically, run the
same off-screen export approach used during development (from the repo root):

.. code-block:: shell

   PYTHONPATH=python python3 - <<'PY'
   import math
   from pathlib import Path
   import imageio.v2 as imageio
   import mujoco
   import numpy as np
   from PIL import Image, ImageDraw, ImageFont
   from mujoco.electrical import SingleEnvSimulation
   from python.examples.electrical.demo_visualizer import (
       _XML_TEMPLATE,
       _LIGHT,
       _HEAVY,
       _UPPER_ARM_MASS,
       _LOWER_ARM_MASS,
       _TARGET_ELBOW,
       _PEAK_TORQUE,
       _MAX_TEMP,
       gravity_feedforward_torque,
   )

   outdir = Path('python/examples/electrical/recordings')
   outdir.mkdir(parents=True, exist_ok=True)
   duration_s = 5.0
   dt = 0.002
   fps_video = 30
   fps_gif = 12
   width, height = 640, 360
   font = ImageFont.load_default()

   def annotate(frame, scenario, state, tagline):
       img = Image.fromarray(frame).convert('RGBA')
       overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
       draw = ImageDraw.Draw(overlay)
       torque = float(state.torques[0])
       current = float(state.currents[0])
       voltage = float(state.bus_voltage)
       power = abs(current * voltage)
       soc = float(state.soc)
       temp = float(state.temperatures[0])
       pos_err = math.degrees(_TARGET_ELBOW - float(state.qpos[1]))
       stall = abs(torque) >= _PEAK_TORQUE * 0.99
       draw.rounded_rectangle((12, 12, 250, 132), radius=10,
                              fill=(0, 0, 0, 165))
       text = (
           f"{scenario.label}\n{tagline}\n"
           f"time  {state.time:4.2f} s\n"
           f"err   {pos_err:+5.1f} deg\n"
           f"I     {current:+6.1f} A\n"
           f"V     {voltage:5.2f} V\n"
           f"SOC   {soc:0.4f}\n"
           f"T     {temp:4.1f} degC"
       )
       draw.multiline_text((22, 20), text, fill=(255, 255, 255, 255), font=font, spacing=3)
       return np.asarray(Image.alpha_composite(img, overlay).convert('RGB'))

   def render_scenario(scenario, stem, tagline):
       sim = SingleEnvSimulation.from_xml(
           _XML_TEMPLATE.format(
               upper_mass=_UPPER_ARM_MASS,
               lower_mass=_LOWER_ARM_MASS,
               payload_mass=scenario.payload_mass,
           ),
           kp=220.0,
           kd=18.0,
       )
       renderer = mujoco.Renderer(sim.model, height, width)
       pos_target = np.array([_TARGET_ELBOW])
       total_steps = int(duration_s / dt)
       total_frames = int(duration_s * fps_video)
       capture_steps = np.linspace(0, total_steps - 1, total_frames).astype(int)
       capture_set = set(int(s) for s in capture_steps)
       frames = []
       for step in range(total_steps):
           ff = gravity_feedforward_torque(
               scenario,
               float(sim.data.qpos[0]),
               float(sim.data.qpos[1]),
           )
           state = sim.step(pos_targets=pos_target, effort_targets=np.array([ff]))
           if step in capture_set:
               renderer.update_scene(sim.data)
               frames.append(annotate(renderer.render(), scenario, state, tagline))
       renderer.close()
       mp4_path = outdir / f'{stem}.mp4'
       gif_path = outdir / f'{stem}.gif'
       with imageio.get_writer(mp4_path, fps=fps_video, codec='libx264', quality=8) as writer:
           for frame in frames:
               writer.append_data(frame)
       gif_indices = np.linspace(0, len(frames) - 1, int(duration_s * fps_gif)).astype(int)
       imageio.mimsave(gif_path, [frames[i] for i in gif_indices], duration=1.0 / fps_gif, loop=0)
       print('wrote', mp4_path)
       print('wrote', gif_path)

   render_scenario(_LIGHT, 'light_acrobot_demo_5s', 'Elbow holds at lower current')
   render_scenario(_HEAVY, 'heavy_acrobot_demo_5s', 'Stall and heating')
   PY

If you have ``ffmpeg`` installed, a good MP4 → GIF conversion command is:

.. code-block:: shell

   ffmpeg -i electrical_demo.mp4 \
     -vf "fps=12,scale=960:-1:flags=lanczos" \
     -loop 0 electrical_demo.gif

For a shorter social-media clip, trim first and then convert:

.. code-block:: shell

   ffmpeg -ss 00:00:02 -t 00:00:06 -i electrical_demo.mp4 \
     -vf "fps=12,scale=960:-1:flags=lanczos" \
     -loop 0 electrical_demo_short.gif

Practical tips:

* Prefer **MP4** for most platforms; it is smaller and looks better than GIF.
* Use **GIF** mainly for short looping previews in chat, issues, or README files.
* A frame rate around ``10--15 fps`` is usually enough for this demo and keeps
  file size manageable.
* The **heavy payload** case is the most visually interesting one to share
  because it clearly shows current saturation, elbow error, and temperature rise.

.. _ElectricalCLI:

Command-line tooling
====================

The ``download_specs`` script lets you pre-fetch and inspect specs
without writing Python code:

.. code-block:: shell

   # Show which remote URL would be used (no download)
   python -m mujoco.electrical.scripts.download_specs --dry-run motor faulhaber_2264w024bp4
   python -m mujoco.electrical.scripts.download_specs --dry-run battery unitree_g1_9ah

   # Download and print a summary
   python -m mujoco.electrical.scripts.download_specs motor faulhaber_2264w024bp4
   python -m mujoco.electrical.scripts.download_specs motor maxon_ec_i_40_488607
   python -m mujoco.electrical.scripts.download_specs battery unitree_g1_9ah

   # Force re-download even if cached
   python -m mujoco.electrical.scripts.download_specs battery unitree_g1_9ah --force

   # List all spec IDs present in ~/.mujoco/cache/
   python -m mujoco.electrical.scripts.download_specs --list-cached motor
   python -m mujoco.electrical.scripts.download_specs --list-cached battery

   # Load from an explicit local file
   python -m mujoco.electrical.scripts.download_specs motor my_id --path /tmp/my_motor.json

Example output for ``faulhaber_2264w024bp4``:

.. code-block:: text

   Loaded motor: faulhaber_2264w024bp4 (Faulhaber 2264W024BP4)
     RL circuit : yes
     Thermal    : yes
     Peak torque: 1.311 N·m
     Voltage    : 0.0–24.0 V

.. _ElectricalPhysics:

Physics model
=============

.. _ElectricalPhysicsMotor:

Motor — full RL-circuit path
-----------------------------

Used when ``spec.has_rl_circuit`` is ``True``
(i.e. both ``resistance`` and ``inductance`` are not ``None``).

Given user-supplied position target :math:`q_d`, velocity target
:math:`\dot{q}_d`, and feed-forward effort :math:`\tau_d`:

.. math::

   \begin{aligned}
   e_\text{back} &= K_e\,\dot{q} \\
   \tau_\text{des} &= k_p(q_d - q) + k_d(\dot{q}_d - \dot{q}) + \tau_d \\
   I_\text{target} &= \tau_\text{des} / K_t \\
   V_\text{terminal} &= I_\text{target}\,R
       + L\,\frac{I_\text{target} - I_\text{prev}}{\Delta t}
       + e_\text{back} \\
   V &= \text{clip}(V_\text{terminal},\; v_\text{min},\; v_\text{bus}) \\
   I_\text{actual} &= \frac{V - e_\text{back} + L\,I_\text{prev}/\Delta t}
       {R + L/\Delta t} \\
   \tau &= \text{clip}(K_t\,I_\text{actual},\; -\tau_\text{peak},\; \tau_\text{peak})
   \end{aligned}

Winding temperature (when ``spec.has_thermal``):

.. math::

   \frac{dT}{dt} = \frac{I_\text{actual}^2\,R
                         - (T - T_\text{amb})/R_{th}}{\tau_{th}}

Motor — degraded path
----------------------

Used when ``spec.has_rl_circuit`` is ``False``
(e.g. ``unitree_a1``, which has ``null`` resistance and inductance in
its asset file):

.. math::

   \begin{aligned}
   \tau &= \text{clip}(\tau_\text{des},\; -\tau_\text{peak},\; \tau_\text{peak}) \\
   I &\approx \tau / K_t
   \end{aligned}

Winding temperature is not updated on the degraded path.

Battery
-------

The battery model tracks state-of-charge (SOC), terminal voltage, and
pack temperature:

.. math::

   \begin{aligned}
   V_{OC} &= N_s \cdot \text{interp}(\text{SOC},\; \text{ocv\_soc},\; \text{ocv\_v}) \\
   m_R &= \text{interp}(\text{SOC},\; \text{r\_soc},\; \text{r\_mult}) \\
   R_\text{int} &= R_\text{base}\,m_R\,(1 + \alpha\,(T - T_\text{amb})) \\
   V_\text{term} &= \text{clip}(V_{OC} - I\,R_\text{int},\; V_\text{min},\; V_\text{max}) \\
   \frac{d\,\text{SOC}}{dt} &= -\frac{I}{C_{Ah}\cdot 3600} \\
   \frac{dT}{dt} &= \frac{I^2\,R_\text{int} - (T - T_\text{amb})/R_{th}}{C_{th}}
   \end{aligned}

where :math:`N_s` is the number of cells in series and :math:`C_{th}` is the
thermal capacity.

.. _ElectricalReferencedSpecs:

Reference assets
================

The following assets are used as reference specifications throughout
the test suite and examples.

Default / gold-standard motor — ``faulhaber_2264w024bp4``
----------------------------------------------------------

All fields fully populated; exercises the **full RL-circuit and
thermal** code path.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Parameter
     - Value
     - Unit
   * - Voltage range
     - 0–24
     - V
   * - Resistance :math:`R`
     - 0.22
     - Ω
   * - Inductance :math:`L`
     - 2.4 × 10\ :sup:`−5`
     - H
   * - :math:`K_t = K_e`
     - 0.0118
     - N·m/A
   * - Peak torque
     - 1.311
     - N·m
   * - Thermal resistance
     - 5.0
     - °C/W
   * - Thermal time constant
     - 950
     - s
   * - Max winding temperature
     - 125
     - °C

**Source**: https://github.com/robomotic/mujoco-motors/blob/master/motor_assets/faulhaber/faulhaber_2264w024bp4.json

Alternative fully-specified motor — ``maxon_ec_i_40_488607``
-------------------------------------------------------------

A second fully-specified motor from Maxon, also available in the
community database.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Parameter
     - Value
     - Unit
   * - Voltage range
     - 0–48
     - V
   * - Resistance :math:`R`
     - 0.994
     - Ω
   * - Inductance :math:`L`
     - 9.95 × 10\ :sup:`−4`
     - H
   * - :math:`K_t = K_e`
     - 0.091
     - N·m/A
   * - Peak torque
     - 2.08
     - N·m
   * - Thermal resistance
     - 8.52
     - °C/W
   * - Thermal time constant
     - 1400
     - s
   * - Max winding temperature
     - 155
     - °C

**Source**: https://github.com/robomotic/mujoco-motors/blob/master/motor_assets/maxon/maxon_ec_i_40_488607.json

Degraded motor — ``unitree_a1``
--------------------------------

``resistance``, ``inductance``, and all thermal fields are ``null``
in this asset.  Used to validate the graceful degraded code path.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Parameter
     - Value
     - Unit
   * - Gear ratio
     - 9.1
     - —
   * - :math:`K_t = K_e`
     - 0.9287
     - N·m/A
   * - Peak torque
     - 33.5
     - N·m
   * - Voltage range
     - 20–40
     - V
   * - Resistance
     - *null*
     - —
   * - Inductance
     - *null*
     - —

**Source**: https://github.com/robomotic/mujoco-motors/blob/master/motor_assets/unitree/unitree_a1.json

Reference battery — ``unitree_g1_9ah``
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Parameter
     - Value
     - Unit
   * - Chemistry
     - Li-ion
     - —
   * - Configuration
     - 6S 1P
     - —
   * - Capacity
     - 9.0
     - Ah
   * - Nominal voltage
     - 21.6
     - V
   * - Internal resistance
     - 0.015
     - Ω
   * - Thermal capacity
     - 1200
     - J/°C
   * - Thermal resistance
     - 8.0
     - °C/W
   * - Max temperature
     - 50
     - °C
   * - OCV curve points
     - 9
     - —
   * - R-SOC curve points
     - 5
     - —

**Source**: https://github.com/robomotic/mujoco-batteries/blob/master/battery_assets/unitree/unitree_g1_9ah.json

