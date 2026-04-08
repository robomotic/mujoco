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
"""Tests for ``_dispatch`` — Phase 4a.

The dispatcher is fully testable without the MuJoCo C extension because
``_set_mujoco_hook`` is a no-op when the extension is absent.  Tests verify:

  * Callbacks are executed when ``_global_dispatch`` is called.
  * Execution order respects ascending priority.
  * ``register`` installs the mujoco hook on first registration.
  * ``unregister`` removes the callback and clears the hook when empty.
  * ``clear`` removes all callbacks at once.
  * Module state is isolated between tests via the ``clean_dispatch`` fixture.
"""

import pytest

from mujoco.electrical import _dispatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_dispatch():
    """Ensure dispatcher registry is empty before and after every test."""
    _dispatch.clear()
    yield
    _dispatch.clear()


@pytest.fixture()
def log():
    """A simple list for recording callback invocation order."""
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cb(name: str, log: list):
    """Return a callback that appends *name* to *log* when called."""
    def cb(model, data):
        log.append(name)
    cb.__name__ = name
    return cb


# ===========================================================================
# Basic callback invocation
# ===========================================================================


class TestGlobalDispatch:
    def test_empty_registry_does_not_raise(self):
        """Calling _global_dispatch with no registered callbacks is a no-op."""
        _dispatch._global_dispatch(None, None)

    def test_single_callback_called(self, log):
        cb = make_cb("A", log)
        _dispatch.register(cb)
        _dispatch._global_dispatch(None, None)
        assert log == ["A"]

    def test_multiple_callbacks_all_called(self, log):
        _dispatch.register(make_cb("A", log))
        _dispatch.register(make_cb("B", log))
        _dispatch._global_dispatch(None, None)
        assert set(log) == {"A", "B"}
        assert len(log) == 2

    def test_callback_receives_model_and_data(self):
        """The (model, data) pair is passed through unchanged."""
        received = []

        def cb(m, d):
            received.append((m, d))

        _dispatch.register(cb)
        sentinel_m = object()
        sentinel_d = object()
        _dispatch._global_dispatch(sentinel_m, sentinel_d)
        assert received == [(sentinel_m, sentinel_d)]


# ===========================================================================
# Priority ordering
# ===========================================================================


class TestPriority:
    def test_lower_priority_called_first(self, log):
        """Priority 0 < 10 so 'first' must be called before 'second'."""
        _dispatch.register(make_cb("first", log), priority=0)
        _dispatch.register(make_cb("second", log), priority=10)
        _dispatch._global_dispatch(None, None)
        assert log == ["first", "second"]

    def test_higher_priority_value_called_last(self, log):
        _dispatch.register(make_cb("low", log), priority=100)
        _dispatch.register(make_cb("high", log), priority=1)
        _dispatch._global_dispatch(None, None)
        assert log == ["high", "low"]

    def test_equal_priority_all_called(self, log):
        """Equal priority callbacks are all invoked (order unspecified)."""
        _dispatch.register(make_cb("X", log), priority=5)
        _dispatch.register(make_cb("Y", log), priority=5)
        _dispatch._global_dispatch(None, None)
        assert set(log) == {"X", "Y"}
        assert len(log) == 2

    def test_negative_priority_called_before_zero(self, log):
        _dispatch.register(make_cb("normal", log), priority=0)
        _dispatch.register(make_cb("urgent", log), priority=-1)
        _dispatch._global_dispatch(None, None)
        assert log[0] == "urgent"

    def test_registration_order_does_not_override_priority(self, log):
        """Even if 'late' is registered first, it should run last."""
        _dispatch.register(make_cb("late", log), priority=99)
        _dispatch.register(make_cb("early", log), priority=1)
        _dispatch._global_dispatch(None, None)
        assert log == ["early", "late"]


# ===========================================================================
# register / unregister
# ===========================================================================


class TestRegisterUnregister:
    def test_registry_empty_after_clear(self):
        _dispatch.register(lambda m, d: None)
        _dispatch.clear()
        assert _dispatch._REGISTERED == []

    def test_unregister_removes_callback(self, log):
        cb = make_cb("A", log)
        _dispatch.register(cb)
        _dispatch.unregister(cb)
        _dispatch._global_dispatch(None, None)
        assert log == []

    def test_unregister_unknown_callback_no_error(self):
        """Unregistering a callback that was never registered is a no-op."""
        _dispatch.unregister(lambda m, d: None)

    def test_unregister_removes_all_occurrences(self, log):
        """If the same callback is registered twice both occurrences are removed."""
        cb = make_cb("dup", log)
        _dispatch.register(cb)
        _dispatch.register(cb)
        _dispatch.unregister(cb)
        _dispatch._global_dispatch(None, None)
        assert log == []

    def test_partial_unregister_leaves_others(self, log):
        cb_a = make_cb("A", log)
        cb_b = make_cb("B", log)
        _dispatch.register(cb_a)
        _dispatch.register(cb_b)
        _dispatch.unregister(cb_a)
        _dispatch._global_dispatch(None, None)
        assert log == ["B"]

    def test_registry_length_tracks_registrations(self):
        _dispatch.register(lambda m, d: None)
        _dispatch.register(lambda m, d: None)
        assert len(_dispatch._REGISTERED) == 2

    def test_register_preserves_priority(self):
        cb = lambda m, d: None  # noqa: E731
        _dispatch.register(cb, priority=42)
        assert _dispatch._REGISTERED[0][0] == 42


# ===========================================================================
# Mujoco hook installation (via monkeypatch)
# ===========================================================================


class TestMujocoHook:
    def test_hook_installed_on_first_register(self, monkeypatch):
        """set_mjcb_control is called once with _global_dispatch."""
        import types
        import sys

        calls = []
        fake_mujoco = types.ModuleType("mujoco")
        fake_mujoco.set_mjcb_control = lambda fn: calls.append(fn)

        original = sys.modules.get("mujoco")
        sys.modules["mujoco"] = fake_mujoco
        try:
            _dispatch.register(lambda m, d: None)
            assert len(calls) == 1
            assert calls[0] is _dispatch._global_dispatch
        finally:
            if original is not None:
                sys.modules["mujoco"] = original
            else:
                del sys.modules["mujoco"]

    def test_hook_cleared_on_last_unregister(self, monkeypatch):
        """set_mjcb_control(None) is called when registry empties."""
        import types
        import sys

        calls = []
        fake_mujoco = types.ModuleType("mujoco")
        fake_mujoco.set_mjcb_control = lambda fn: calls.append(fn)

        original = sys.modules.get("mujoco")
        sys.modules["mujoco"] = fake_mujoco
        try:
            cb = lambda m, d: None  # noqa: E731
            _dispatch.register(cb)
            _dispatch.unregister(cb)
            # Last call should be None (hook cleared)
            assert calls[-1] is None
        finally:
            if original is not None:
                sys.modules["mujoco"] = original
            else:
                del sys.modules["mujoco"]

    def test_hook_not_reinstalled_on_second_register(self, monkeypatch):
        """set_mjcb_control is only called on the FIRST registration."""
        import types
        import sys

        install_count = []
        fake_mujoco = types.ModuleType("mujoco")
        fake_mujoco.set_mjcb_control = lambda fn: (
            install_count.append(1) if fn is not None else None
        )

        original = sys.modules.get("mujoco")
        sys.modules["mujoco"] = fake_mujoco
        try:
            _dispatch.register(lambda m, d: None)
            _dispatch.register(lambda m, d: None)
            assert sum(install_count) == 1
        finally:
            if original is not None:
                sys.modules["mujoco"] = original
            else:
                del sys.modules["mujoco"]

    def test_no_attribute_no_error(self, monkeypatch):
        """Stubs without set_mjcb_control don't cause AttributeError."""
        import types
        import sys

        fake_mujoco = types.ModuleType("mujoco")
        # deliberately no set_mjcb_control attribute

        original = sys.modules.get("mujoco")
        sys.modules["mujoco"] = fake_mujoco
        try:
            # should not raise
            _dispatch.register(lambda m, d: None)
            _dispatch.clear()
        finally:
            if original is not None:
                sys.modules["mujoco"] = original
            else:
                del sys.modules["mujoco"]
