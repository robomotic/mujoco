// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PLUGIN_ACTUATOR_ELECTRICAL_MOTOR_H_
#define MUJOCO_PLUGIN_ACTUATOR_ELECTRICAL_MOTOR_H_

#include <memory>
#include <string>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

namespace mujoco::plugin::actuator {

// ---------------------------------------------------------------------------
// Activation-variable slot indices stored in d->act.
// Each plugin instance owns 2 act variables per actuator:
//   [actadr + kActI]    = previous winding current I_prev (A)
//   [actadr + kActTemp] = winding temperature (°C)
// ---------------------------------------------------------------------------
constexpr int kActI    = 0;
constexpr int kActTemp = 1;
constexpr int kActDim  = 2;  // total act slots per actuator

// ---------------------------------------------------------------------------
// ElectricalMotorConfig
// ---------------------------------------------------------------------------
// Holds all per-instance parameters loaded once during Init.  The motor JSON
// spec is resolved via the following priority (mirrors the Python stack):
//   1. spec_path plugin attribute (explicit path to JSON file)
//   2. MUJOCO_MOTOR_PATH environment variable (colon-separated search dirs)
//   3. ~/.mujoco/motors/{vendor}/{id}.json
//   4. Inline defaults (when the JSON cannot be found — returns false from
//      LoadFromFile and the caller issues a warning).
// ---------------------------------------------------------------------------
struct ElectricalMotorConfig {
  // ---- Plugin attributes ----
  std::string motor_id;
  double kp = 100.0;
  double kd = 10.0;
  std::string spec_path;  // optional override path to .json

  // ---- Loaded spec fields ----
  double Kt  = 1.0;    // N·m/A
  double Ke  = 1.0;    // V·s/rad
  double R   = 0.0;    // Ω
  double L   = 0.0;    // H
  double v_min = 0.0;  // V
  double v_max = 24.0; // V
  double peak_torque = 1.0;  // N·m

  // ---- Thermal ----
  double R_th  = 0.0;   // °C/W
  double tau_th = 1.0;  // s
  double T_amb = 25.0;  // °C

  // ---- Path flags ----
  bool has_rl      = false;
  bool has_thermal = false;

  // Reads plugin attributes and populates all fields.
  // Returns false and logs a warning if the motor JSON cannot be loaded.
  static std::unique_ptr<ElectricalMotorConfig> Create(
      const mjModel* m, int instance);
};

// ---------------------------------------------------------------------------
// ElectricalMotorPlugin
// ---------------------------------------------------------------------------
class ElectricalMotorPlugin {
 public:
  // Factory — returns nullptr on misconfiguration.
  static std::unique_ptr<ElectricalMotorPlugin> Create(
      const mjModel* m, int instance);

  // Number of act variables per actuator controlled by this instance.
  static int StateSize(const mjModel* m, int instance);

  // Reset per-actuator activation state to initial conditions.
  void Reset(mjtNum* plugin_state);

  // Compute act_dot (Euler derivative for I_prev and temperature).
  void ActDot(const mjModel* m, mjData* d, int instance) const;

  // Compute actuator_force for all actuators in this instance.
  void Compute(const mjModel* m, mjData* d, int instance);

  // Advance plugin state (act variables already updated by MuJoCo).
  void Advance(const mjModel* m, mjData* d, int instance) const;

  // Adds the plugin to the global MuJoCo plugin registry.
  static void RegisterPlugin();

 private:
  explicit ElectricalMotorPlugin(
      std::unique_ptr<ElectricalMotorConfig> config,
      std::vector<int> actuators);

  // Compute torque and current for one actuator using the full RL circuit.
  static void ComputeRL(
      const ElectricalMotorConfig& cfg, double pos, double vel,
      double pos_tgt, double vel_tgt, double I_prev, double dt,
      double& torque_out, double& current_out);

  // Compute torque and current for one actuator using the degraded path.
  static void ComputeDegraded(
      const ElectricalMotorConfig& cfg, double pos,
      double pos_tgt, double& torque_out, double& current_out);

  std::unique_ptr<ElectricalMotorConfig> config_;
  std::vector<int> actuators_;
};

}  // namespace mujoco::plugin::actuator

#endif  // MUJOCO_PLUGIN_ACTUATOR_ELECTRICAL_MOTOR_H_
