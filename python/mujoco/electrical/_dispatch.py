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
"""Module-level ``mjcb_control`` priority dispatcher.

All :class:`~mujoco.electrical.simulation.SingleEnvSimulation` instances
share a single ``mujoco.set_mjcb_control`` hook.  This module manages a
sorted list of callbacks and fans out to them in ascending priority order
(lower number = called first).

Usage example::

    from mujoco.electrical._dispatch import register, unregister

    def my_control(model, data):
        data.ctrl[0] = 1.0

    register(my_control, priority=0)
    # ... simulation runs ...
    unregister(my_control)

The ``mujoco.set_mjcb_control`` hook is **installed** on the first
registration and **cleared** when the registry becomes empty.  The module
gracefully no-ops when the MuJoCo C extension is not available (useful in
test environments).
"""

from __future__ import annotations

from typing import Callable

# Module-level registry: list of (priority, callback) pairs.
# Lower priority value = called earlier.
_REGISTERED: list[tuple[int, Callable]] = []


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #

def _global_dispatch(model, data) -> None:
  """Fan-out callback installed as ``mujoco.set_mjcb_control``.

  Calls all registered handlers in ascending priority order.
  """
  for _, cb in sorted(_REGISTERED, key=lambda x: x[0]):
    cb(model, data)


def _set_mujoco_hook(fn) -> None:
  """Install or clear the mujoco control callback.

  No-ops silently when the C extension is absent or does not expose
  ``set_mjcb_control`` (e.g. in unit-test stubs).
  """
  try:
    import mujoco  # noqa: PLC0415
    if hasattr(mujoco, "set_mjcb_control"):
      mujoco.set_mjcb_control(fn)
  except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

def register(cb: Callable, priority: int = 0) -> None:
  """Register a control callback.

  Args:
    cb:       Callable with signature ``(model, data) -> None``.
    priority: Execution order. Lower value = called first (default 0).

  Note:
    Re-registering the same callable schedules it to run multiple times.
    Call :func:`unregister` to remove it.  The ``mujoco.set_mjcb_control``
    hook is installed on the first registration.
  """
  _REGISTERED.append((priority, cb))
  if len(_REGISTERED) == 1:
    _set_mujoco_hook(_global_dispatch)


def unregister(cb: Callable) -> None:
  """Remove every occurrence of *cb* from the registry.

  Clears the mujoco hook when the registry becomes empty.
  """
  _REGISTERED[:] = [(p, c) for p, c in _REGISTERED if c is not cb]
  if not _REGISTERED:
    _set_mujoco_hook(None)


def clear() -> None:
  """Remove all registered callbacks and uninstall the mujoco hook."""
  _REGISTERED.clear()
  _set_mujoco_hook(None)
