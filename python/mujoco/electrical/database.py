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
"""Motor and battery specification databases with remote-fetch and MD5 caching.

Search priority (highest → lowest) for both databases:

  1. Explicit ``path`` argument to :meth:`load`.
  2. Colon-separated directories in ``MUJOCO_MOTOR_PATH`` /
     ``MUJOCO_BATTERY_PATH`` environment variables.
  3. ``~/.mujoco/motors/`` or ``~/.mujoco/batteries/`` user directories.
  4. Remote community repository — downloaded and cached in
     ``~/.mujoco/cache/`` with MD5 verification.

Remote URL conventions
----------------------
Motor  : ``MOTOR_REMOTE_BASE/{vendor}/{motor_id}.json``
Battery: ``BATTERY_REMOTE_BASE/{vendor}/{battery_id}.json``

The vendor prefix is derived by splitting the ID on the first underscore
(e.g. ``faulhaber_2264w024bp4`` → vendor ``faulhaber``).  A GitHub file-tree
fallback is used if the direct URL returns 404.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import urllib.error
import urllib.request
from typing import Optional

from mujoco.electrical.battery_spec import BatterySpecification
from mujoco.electrical.motor_spec import MotorSpecification

# --------------------------------------------------------------------------- #
# Remote repository URLs                                                        #
# --------------------------------------------------------------------------- #
_MOTOR_REMOTE_BASE = (
    "https://raw.githubusercontent.com/robomotic/mujoco-motors/master"
    "/motor_assets"
)
_BATTERY_REMOTE_BASE = (
    "https://raw.githubusercontent.com/robomotic/mujoco-batteries/master"
    "/battery_assets"
)

# GitHub API endpoint used as a fallback index when motor vendor cannot be
# inferred from the ID.
_MOTOR_TREE_API = (
    "https://api.github.com/repos/robomotic/mujoco-motors/git/trees"
    "/master?recursive=1"
)

# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #

def _cache_dir() -> pathlib.Path:
  d = pathlib.Path.home() / ".mujoco" / "cache"
  d.mkdir(parents=True, exist_ok=True)
  return d


def _md5(data: bytes) -> str:
  return hashlib.md5(data).hexdigest()  # noqa: S324 — MD5 used for cache, not security


def _fetch_url(url: str) -> bytes:
  """Download *url* and return raw bytes.  Raises ``urllib.error.URLError``."""
  req = urllib.request.Request(
      url,
      headers={"User-Agent": "mujoco-electrical/1.0"},
  )
  with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
    return resp.read()


def _cached_fetch(url: str, force: bool = False) -> bytes:
  """Download *url* once and cache the result locally.

  The cache key is the MD5 of the URL string.  The cached file is always
  trusted on subsequent calls unless *force* is ``True``.  A ``<key>.md5``
  sidecar stores the content MD5 so callers can verify integrity without
  re-downloading.

  Parameters
  ----------
  url:
      Remote URL to fetch.
  force:
      When ``True``, bypass the cache and re-download unconditionally.
  """
  cache = _cache_dir()
  url_hash = _md5(url.encode())
  cached_file = cache / f"{url_hash}.json"
  cached_md5 = cache / f"{url_hash}.md5"

  if cached_file.exists() and not force:
    return cached_file.read_bytes()

  data = _fetch_url(url)
  cached_file.write_bytes(data)
  cached_md5.write_text(_md5(data))
  return data


def _vendor_from_id(spec_id: str) -> str:
  """Return the vendor prefix from a spec ID.

  Convention: first underscore-delimited token is the vendor.
  Example: 'faulhaber_2264w024bp4' → 'faulhaber'.
  """
  return spec_id.split("_")[0]


def _find_motor_url_via_tree(motor_id: str) -> Optional[str]:
  """Use the GitHub tree API to locate a motor JSON by filename."""
  try:
    data = json.loads(_fetch_url(_MOTOR_TREE_API))
    filename = f"{motor_id}.json"
    for item in data.get("tree", []):
      path: str = item.get("path", "")
      if path.endswith(f"/{filename}") or path == filename:
        vendor_path = path  # e.g. "motor_assets/faulhaber/faulhaber_2264w024bp4.json"
        return (
            "https://raw.githubusercontent.com/robomotic/mujoco-motors"
            f"/master/{vendor_path}"
        )
  except (urllib.error.URLError, KeyError, json.JSONDecodeError):
    pass
  return None


# --------------------------------------------------------------------------- #
# MotorDatabase                                                                 #
# --------------------------------------------------------------------------- #

class MotorDatabase:
  """Loads :class:`~mujoco.electrical.motor_spec.MotorSpecification` objects.

  Usage::

      db = MotorDatabase()
      spec = db.load("faulhaber_2264w024bp4")
  """

  def load(
      self,
      motor_id: str,
      path: Optional[str] = None,
      force: bool = False,
  ) -> MotorSpecification:
    """Load a motor specification by ID.

    Parameters
    ----------
    motor_id:
        Unique motor identifier, e.g. ``'faulhaber_2264w024bp4'``.
    path:
        Optional explicit file-system path to a JSON file.  When provided
        all other search locations are skipped.
    force:
        Re-download from remote even if a cached copy exists.

    Returns
    -------
    MotorSpecification

    Raises
    ------
    FileNotFoundError
        When the spec cannot be found locally or fetched remotely.
    """
    data = self._load_json(motor_id, path, force=force)
    return MotorSpecification.from_dict(data)

  def search(self, query: str) -> list[MotorSpecification]:
    """Return all locally-cached specs whose ID contains *query*."""
    results: list[MotorSpecification] = []
    for json_path in self._all_local_paths():
      try:
        with open(json_path) as f:
          data = json.load(f)
        if query.lower() in data.get("motor_id", "").lower():
          results.append(MotorSpecification.from_dict(data))
      except (OSError, json.JSONDecodeError, TypeError):
        continue
    return results

  def list_cached(self) -> list[str]:
    """Return motor IDs present in ``~/.mujoco/cache/``."""
    ids: list[str] = []
    for p in (_cache_dir()).glob("*.json"):
      try:
        with open(p) as f:
          data = json.load(f)
        if "motor_id" in data:
          ids.append(data["motor_id"])
      except (OSError, json.JSONDecodeError):
        continue
    return ids

  # ------------------------------------------------------------------ #
  # Internal                                                             #
  # ------------------------------------------------------------------ #

  def _load_json(
      self,
      motor_id: str,
      explicit_path: Optional[str],
      force: bool = False,
  ) -> dict:
    # 1. Explicit path.
    if explicit_path is not None:
      with open(explicit_path) as f:
        return json.load(f)

    # 2. MUJOCO_MOTOR_PATH env var.
    for directory in self._env_paths("MUJOCO_MOTOR_PATH"):
      candidate = pathlib.Path(directory) / f"{motor_id}.json"
      if candidate.is_file():
        with open(candidate) as f:
          return json.load(f)

    # 3. ~/.mujoco/motors/
    user_dir = pathlib.Path.home() / ".mujoco" / "motors"
    candidate = user_dir / f"{motor_id}.json"
    if candidate.is_file():
      with open(candidate) as f:
        return json.load(f)

    # 4. Remote fetch with cache.
    return self._fetch_remote(motor_id, force=force)

  def _fetch_remote(self, motor_id: str, force: bool = False) -> dict:
    vendor = _vendor_from_id(motor_id)
    primary_url = f"{_MOTOR_REMOTE_BASE}/{vendor}/{motor_id}.json"

    try:
      raw = _cached_fetch(primary_url, force=force)
      return json.loads(raw)
    except urllib.error.HTTPError as exc:
      if exc.code != 404:
        raise

    # Vendor-prefix inference failed — try the GitHub tree API.
    fallback_url = _find_motor_url_via_tree(motor_id)
    if fallback_url is not None:
      try:
        raw = _cached_fetch(fallback_url, force=force)
        return json.loads(raw)
      except urllib.error.URLError:
        pass

    raise FileNotFoundError(
        f"Motor spec '{motor_id}' not found locally or at {primary_url}. "
        "Set MUJOCO_MOTOR_PATH or place the JSON in ~/.mujoco/motors/."
    )

  @staticmethod
  def _env_paths(var: str) -> list[str]:
    value = os.environ.get(var, "")
    return [p for p in value.split(":") if p]

  def _all_local_paths(self) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for d in self._env_paths("MUJOCO_MOTOR_PATH"):
      paths.extend(pathlib.Path(d).glob("*.json"))
    user_dir = pathlib.Path.home() / ".mujoco" / "motors"
    if user_dir.is_dir():
      paths.extend(user_dir.glob("*.json"))
    return paths


# --------------------------------------------------------------------------- #
# BatteryDatabase                                                               #
# --------------------------------------------------------------------------- #

class BatteryDatabase:
  """Loads :class:`~mujoco.electrical.battery_spec.BatterySpecification` objects.

  Usage::

      db = BatteryDatabase()
      spec = db.load("unitree_g1_9ah")
  """

  def load(
      self,
      battery_id: str,
      path: Optional[str] = None,
      force: bool = False,
  ) -> BatterySpecification:
    """Load a battery specification by ID.

    Parameters
    ----------
    battery_id:
        Unique battery identifier, e.g. ``'unitree_g1_9ah'``.
    path:
        Optional explicit file-system path to a JSON file.
    force:
        Re-download from remote even if a cached copy exists.

    Returns
    -------
    BatterySpecification

    Raises
    ------
    FileNotFoundError
        When the spec cannot be found locally or fetched remotely.
    """
    data = self._load_json(battery_id, path, force=force)
    return BatterySpecification.from_dict(data)

  def search(self, query: str) -> list[BatterySpecification]:
    """Return all locally-cached specs whose ID contains *query*."""
    results: list[BatterySpecification] = []
    for json_path in self._all_local_paths():
      try:
        with open(json_path) as f:
          data = json.load(f)
        if query.lower() in data.get("battery_id", "").lower():
          results.append(BatterySpecification.from_dict(data))
      except (OSError, json.JSONDecodeError, TypeError):
        continue
    return results

  def list_cached(self) -> list[str]:
    """Return battery IDs present in ``~/.mujoco/cache/``."""
    ids: list[str] = []
    for p in (_cache_dir()).glob("*.json"):
      try:
        with open(p) as f:
          data = json.load(f)
        if "battery_id" in data:
          ids.append(data["battery_id"])
      except (OSError, json.JSONDecodeError):
        continue
    return ids

  # ------------------------------------------------------------------ #
  # Internal                                                             #
  # ------------------------------------------------------------------ #

  def _load_json(
      self,
      battery_id: str,
      explicit_path: Optional[str],
      force: bool = False,
  ) -> dict:
    # 1. Explicit path.
    if explicit_path is not None:
      with open(explicit_path) as f:
        return json.load(f)

    # 2. MUJOCO_BATTERY_PATH env var.
    for directory in self._env_paths("MUJOCO_BATTERY_PATH"):
      candidate = pathlib.Path(directory) / f"{battery_id}.json"
      if candidate.is_file():
        with open(candidate) as f:
          return json.load(f)

    # 3. ~/.mujoco/batteries/
    user_dir = pathlib.Path.home() / ".mujoco" / "batteries"
    candidate = user_dir / f"{battery_id}.json"
    if candidate.is_file():
      with open(candidate) as f:
        return json.load(f)

    # 4. Remote fetch with cache.
    return self._fetch_remote(battery_id, force=force)

  def _fetch_remote(self, battery_id: str, force: bool = False) -> dict:
    vendor = _vendor_from_id(battery_id)
    url = f"{_BATTERY_REMOTE_BASE}/{vendor}/{battery_id}.json"
    try:
      raw = _cached_fetch(url, force=force)
      return json.loads(raw)
    except urllib.error.URLError as exc:
      raise FileNotFoundError(
          f"Battery spec '{battery_id}' not found locally or at {url}. "
          "Set MUJOCO_BATTERY_PATH or place the JSON in ~/.mujoco/batteries/."
      ) from exc

  @staticmethod
  def _env_paths(var: str) -> list[str]:
    value = os.environ.get(var, "")
    return [p for p in value.split(":") if p]

  def _all_local_paths(self) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for d in self._env_paths("MUJOCO_BATTERY_PATH"):
      paths.extend(pathlib.Path(d).glob("*.json"))
    user_dir = pathlib.Path.home() / ".mujoco" / "batteries"
    if user_dir.is_dir():
      paths.extend(user_dir.glob("*.json"))
    return paths
