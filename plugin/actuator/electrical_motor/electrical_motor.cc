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

#include "electrical_motor.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Simple JSON parsing: we only need a handful of scalar fields.
// This avoids pulling in an external JSON library.
#include <mujoco/mujoco.h>

namespace mujoco::plugin::actuator {
namespace {

// ---------------------------------------------------------------------------
// Minimal JSON field extraction
// ---------------------------------------------------------------------------
// Extracts the numeric value for a JSON key of the form  "key": <number|null>
// Returns true and sets *out on success; returns false if absent or null.
bool JsonGetDouble(const std::string& json, const char* key, double* out) {
  // Find "key":
  std::string search = std::string("\"") + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) return false;
  pos += search.size();
  // Skip whitespace
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (pos >= json.size()) return false;
  // Check for null
  if (json.compare(pos, 4, "null") == 0) return false;
  // Parse number
  char* end = nullptr;
  double val = std::strtod(json.c_str() + pos, &end);
  if (end == json.c_str() + pos) return false;
  *out = val;
  return true;
}

// ---------------------------------------------------------------------------
// JSON array: read first element of "voltage_range": [v_min, v_max]
// ---------------------------------------------------------------------------
bool JsonGetVoltageRange(const std::string& json, double* v_min, double* v_max) {
  size_t pos = json.find("\"voltage_range\":");
  if (pos == std::string::npos) return false;
  pos = json.find('[', pos);
  if (pos == std::string::npos) return false;
  ++pos;
  char* end = nullptr;
  *v_min = std::strtod(json.c_str() + pos, &end);
  if (end == json.c_str() + pos) return false;
  pos = json.find(',', static_cast<size_t>(end - json.c_str()));
  if (pos == std::string::npos) return false;
  ++pos;
  *v_max = std::strtod(json.c_str() + pos, nullptr);
  return true;
}

// ---------------------------------------------------------------------------
// Spec file resolution: returns full path or empty string on failure.
// Priority:
//   1. spec_path attribute (non-empty)
//   2. MUJOCO_MOTOR_PATH env-var directories
//   3. ~/.mujoco/motors/{vendor}/{id}.json
// ---------------------------------------------------------------------------
std::string VendorFromId(const std::string& motor_id) {
  size_t pos = motor_id.find('_');
  if (pos == std::string::npos) return motor_id;
  return motor_id.substr(0, pos);
}

std::string ResolveSpecPath(const std::string& motor_id,
                             const std::string& spec_path) {
  // 1. Explicit path
  if (!spec_path.empty()) return spec_path;

  std::string vendor = VendorFromId(motor_id);
  std::string filename = motor_id + ".json";

  // Helper: try a single directory
  auto try_dir = [&](const std::string& dir) -> std::string {
    std::string candidate = dir;
    if (!candidate.empty() && candidate.back() != '/') candidate += '/';
    candidate += vendor + "/" + filename;
    std::ifstream f(candidate);
    if (f.good()) return candidate;
    // Also try without vendor subdirectory
    candidate = dir;
    if (!candidate.empty() && candidate.back() != '/') candidate += '/';
    candidate += filename;
    f.open(candidate);
    if (f.good()) return candidate;
    return "";
  };

  // 2. MUJOCO_MOTOR_PATH (colon-separated)
  const char* env = std::getenv("MUJOCO_MOTOR_PATH");
  if (env != nullptr) {
    std::istringstream ss(env);
    std::string dir;
    while (std::getline(ss, dir, ':')) {
      std::string found = try_dir(dir);
      if (!found.empty()) return found;
    }
  }

  // 3. ~/.mujoco/motors/
  const char* home = std::getenv("HOME");
  if (home != nullptr) {
    std::string motors_dir = std::string(home) + "/.mujoco/motors";
    std::string found = try_dir(motors_dir);
    if (!found.empty()) return found;
  }

  return "";
}

// Reads entire file into a string; returns false on failure.
bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

// Reads plugin attribute string; returns empty string if absent.
std::string GetStringAttr(const mjModel* m, int instance, const char* key) {
  const char* val = mj_getPluginConfig(m, instance, key);
  if (val == nullptr) return "";
  return val;
}

double GetDoubleAttr(const mjModel* m, int instance, const char* key,
                     double default_val) {
  const char* val = mj_getPluginConfig(m, instance, key);
  if (val == nullptr || val[0] == '\0') return default_val;
  return std::strtod(val, nullptr);
}

}  // namespace

// ---------------------------------------------------------------------------
// ElectricalMotorConfig::Create
// ---------------------------------------------------------------------------
std::unique_ptr<ElectricalMotorConfig> ElectricalMotorConfig::Create(
    const mjModel* m, int instance) {
  auto cfg = std::make_unique<ElectricalMotorConfig>();

  cfg->motor_id  = GetStringAttr(m, instance, "motor_id");
  cfg->kp        = GetDoubleAttr(m, instance, "kp",  100.0);
  cfg->kd        = GetDoubleAttr(m, instance, "kd",  10.0);
  cfg->spec_path = GetStringAttr(m, instance, "spec_path");

  if (cfg->motor_id.empty()) {
    mju_warning("electrical_motor: 'motor_id' attribute is required");
    return nullptr;
  }

  std::string path = ResolveSpecPath(cfg->motor_id, cfg->spec_path);
  if (path.empty()) {
    mju_warning(
        "electrical_motor: spec file for '%s' not found. "
        "Set MUJOCO_MOTOR_PATH or use spec_path attribute.",
        cfg->motor_id.c_str());
    return nullptr;
  }

  std::string json;
  if (!ReadFile(path, &json)) {
    mju_warning("electrical_motor: could not read '%s'", path.c_str());
    return nullptr;
  }

  // Required fields
  if (!JsonGetDouble(json, "motor_constant_kt", &cfg->Kt) ||
      !JsonGetDouble(json, "motor_constant_ke", &cfg->Ke) ||
      !JsonGetDouble(json, "peak_torque",        &cfg->peak_torque)) {
    mju_warning(
        "electrical_motor: missing required fields in '%s'", path.c_str());
    return nullptr;
  }

  if (!JsonGetVoltageRange(json, &cfg->v_min, &cfg->v_max)) {
    cfg->v_min = 0.0;
    cfg->v_max = 24.0;
  }

  // Optional RL fields
  double R_val = 0.0, L_val = 0.0;
  bool has_R = JsonGetDouble(json, "resistance", &R_val);
  bool has_L = JsonGetDouble(json, "inductance", &L_val);
  if (has_R && has_L) {
    cfg->R      = R_val;
    cfg->L      = L_val;
    cfg->has_rl = true;
  }

  // Optional thermal fields
  double R_th = 0.0, tau_th = 0.0, T_amb = 25.0;
  bool has_Rth  = JsonGetDouble(json, "thermal_resistance",    &R_th);
  bool has_tau  = JsonGetDouble(json, "thermal_time_constant", &tau_th);
  JsonGetDouble(json, "ambient_temperature", &T_amb);
  if (has_Rth && has_tau) {
    cfg->R_th        = R_th;
    cfg->tau_th      = tau_th;
    cfg->T_amb       = T_amb;
    cfg->has_thermal = true;
  }

  // Optional drive current limit (stall_current = peak current from drive electronics)
  double i_max_val = 0.0;
  if (JsonGetDouble(json, "stall_current", &i_max_val) && i_max_val > 0.0) {
    cfg->i_max              = i_max_val;
    cfg->has_current_limit  = true;
  }

  return cfg;
}

// ---------------------------------------------------------------------------
// ElectricalMotorPlugin
// ---------------------------------------------------------------------------
ElectricalMotorPlugin::ElectricalMotorPlugin(
    std::unique_ptr<ElectricalMotorConfig> config,
    std::vector<int> actuators)
    : config_(std::move(config)), actuators_(std::move(actuators)) {}

std::unique_ptr<ElectricalMotorPlugin> ElectricalMotorPlugin::Create(
    const mjModel* m, int instance) {
  auto config = ElectricalMotorConfig::Create(m, instance);
  if (config == nullptr) return nullptr;

  std::vector<int> actuators;
  for (int i = 0; i < m->nu; i++) {
    if (m->actuator_plugin[i] == instance) {
      actuators.push_back(i);
    }
  }
  if (actuators.empty()) {
    mju_warning("electrical_motor: no actuators found for instance %d",
                instance);
    return nullptr;
  }

  return std::unique_ptr<ElectricalMotorPlugin>(
      new ElectricalMotorPlugin(std::move(config), std::move(actuators)));
}

int ElectricalMotorPlugin::StateSize(const mjModel* /*m*/, int /*instance*/) {
  return 0;  // plugin_state unused; I_prev and temp live in d->act
}

void ElectricalMotorPlugin::Reset(mjtNum* /*plugin_state*/) {}

// ---------------------------------------------------------------------------
// Physics helpers
// ---------------------------------------------------------------------------
void ElectricalMotorPlugin::ComputeRL(
    const ElectricalMotorConfig& cfg, double pos, double vel,
    double pos_tgt, double vel_tgt, double I_prev, double dt,
    double& torque_out, double& current_out) {
  double back_emf   = cfg.Ke * vel;
  double effort_des = cfg.kp * (pos_tgt - pos) + cfg.kd * (vel_tgt - vel);
  double I_target   = effort_des / cfg.Kt;
  double V_term     = I_target * cfg.R
                    + cfg.L * (I_target - I_prev) / dt
                    + back_emf;
  double voltage    = mjMAX(cfg.v_min, mjMIN(cfg.v_max, V_term));
  double I_actual   = (voltage - back_emf + cfg.L * I_prev / dt)
                    / (cfg.R + cfg.L / dt);
  if (cfg.has_current_limit) {
    I_actual = mjMAX(-cfg.i_max, mjMIN(cfg.i_max, I_actual));
  }
  torque_out  = mjMAX(-cfg.peak_torque,
                      mjMIN(cfg.peak_torque, cfg.Kt * I_actual));
  current_out = I_actual;
}

void ElectricalMotorPlugin::ComputeDegraded(
    const ElectricalMotorConfig& cfg, double pos, double pos_tgt,
    double& torque_out, double& current_out) {
  double effort_des = cfg.kp * (pos_tgt - pos);
  torque_out  = mjMAX(-cfg.peak_torque, mjMIN(cfg.peak_torque, effort_des));
  current_out = torque_out / cfg.Kt;
  if (cfg.has_current_limit) {
    current_out = mjMAX(-cfg.i_max, mjMIN(cfg.i_max, current_out));
    torque_out  = current_out * cfg.Kt;
  }
}

// ---------------------------------------------------------------------------
// ActDot: set rate-of-change for I_prev and temperature act variables.
// ---------------------------------------------------------------------------
void ElectricalMotorPlugin::ActDot(
    const mjModel* m, mjData* d, int /*instance*/) const {
  const auto& cfg = *config_;
  double dt = m->opt.timestep;

  for (int adr : actuators_) {
    int act_base = m->actuator_actadr[adr];
    double I_prev = d->act[act_base + kActI];
    double T_prev = d->act[act_base + kActTemp];

    double pos     = d->actuator_length[adr];
    double vel     = d->actuator_velocity[adr];
    double pos_tgt = d->ctrl[adr];

    double torque = 0.0, current = 0.0;
    if (cfg.has_rl) {
      ComputeRL(cfg, pos, vel, pos_tgt, /*vel_tgt=*/0.0, I_prev, dt,
                torque, current);
    } else {
      ComputeDegraded(cfg, pos, pos_tgt, torque, current);
    }

    // I_prev update: act_dot[kActI] = (current - I_prev) / dt (Euler)
    d->act_dot[act_base + kActI] = (current - I_prev) / dt;

    // Temperature update: act_dot[kActTemp] = dT/dt
    if (cfg.has_thermal && cfg.has_rl) {
      double tau_th = mju_max(cfg.tau_th, mjMINVAL);
      double dT_dt  = (current * current * cfg.R
                       - (T_prev - cfg.T_amb) / cfg.R_th)
                    / tau_th;
      d->act_dot[act_base + kActTemp] = dT_dt;
    } else {
      d->act_dot[act_base + kActTemp] = 0.0;
    }
  }
}

// ---------------------------------------------------------------------------
// Compute: write actuator_force for all actuators.
// ---------------------------------------------------------------------------
void ElectricalMotorPlugin::Compute(
    const mjModel* m, mjData* d, int /*instance*/) {
  const auto& cfg = *config_;
  double dt = m->opt.timestep;

  for (int adr : actuators_) {
    int act_base = m->actuator_actadr[adr];
    double I_prev  = d->act[act_base + kActI];

    double pos     = d->actuator_length[adr];
    double vel     = d->actuator_velocity[adr];
    double pos_tgt = d->ctrl[adr];

    double torque = 0.0, current = 0.0;
    if (cfg.has_rl) {
      ComputeRL(cfg, pos, vel, pos_tgt, /*vel_tgt=*/0.0, I_prev, dt,
                torque, current);
    } else {
      ComputeDegraded(cfg, pos, pos_tgt, torque, current);
    }

    d->actuator_force[adr] = torque;
  }
}

void ElectricalMotorPlugin::Advance(
    const mjModel* /*m*/, mjData* /*d*/, int /*instance*/) const {
  // act variables updated by MuJoCo integrating act_dot.
}

// ---------------------------------------------------------------------------
// RegisterPlugin
// ---------------------------------------------------------------------------
void ElectricalMotorPlugin::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name           = "mujoco.actuator.electrical_motor";
  plugin.capabilityflags = mjPLUGIN_ACTUATOR;

  // Declared attributes (must match XML <config> keys)
  static const char* kAttributes[] = {
      "motor_id", "kp", "kd", "spec_path"};
  plugin.nattribute = 4;
  plugin.attributes = kAttributes;

  // Each actuator controlled by this plugin needs kActDim act variables.
  plugin.nstate = +[](const mjModel* m, int instance) -> int {
    return ElectricalMotorPlugin::StateSize(m, instance);
  };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) -> int {
    auto em = ElectricalMotorPlugin::Create(m, instance);
    if (em == nullptr) return -1;

    // Validate actnum for every controlled actuator
    for (int i = 0; i < m->nu; i++) {
      if (m->actuator_plugin[i] != instance) continue;
      if (m->actuator_actnum[i] != kActDim) {
        mju_warning(
            "electrical_motor: actuator %d must have actdim=\"%d\"; "
            "add actdim=\"%d\" to the actuator element.",
            i, kActDim, kActDim);
        return -1;
      }
    }

    d->plugin_data[instance] =
        reinterpret_cast<uintptr_t>(em.release());
    return 0;
  };

  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<ElectricalMotorPlugin*>(
        d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };

  plugin.reset = +[](const mjModel* /*m*/, mjtNum* plugin_state,
                     void* plugin_data, int /*instance*/) {
    auto* em =
        reinterpret_cast<ElectricalMotorPlugin*>(plugin_data);
    em->Reset(plugin_state);
  };

  plugin.actuator_act_dot =
      +[](const mjModel* m, mjData* d, int instance) {
        auto* em = reinterpret_cast<ElectricalMotorPlugin*>(
            d->plugin_data[instance]);
        em->ActDot(m, d, instance);
      };

  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance,
          int /* capability_bit */) {
        auto* em = reinterpret_cast<ElectricalMotorPlugin*>(
            d->plugin_data[instance]);
        em->Compute(m, d, instance);
      };

  plugin.advance = +[](const mjModel* m, mjData* d, int instance) {
    auto* em = reinterpret_cast<ElectricalMotorPlugin*>(
        d->plugin_data[instance]);
    em->Advance(m, d, instance);
  };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::actuator
