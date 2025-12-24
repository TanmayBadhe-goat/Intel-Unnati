# %% [markdown]
# # Edge-Ready Smart Digital Twin for Predictive Maintenance of Industrial Motor Drive Systems
# ## using Reduced-Order Modelling and AI
#
# This notebook implements a software-only Smart Digital Twin aligned with
# **Intel Unnati Problem Statement 5**.
#
# **Pipeline:**
# Virtual Motor (FOM) → Sensors → Reduced Order Model (ROM)
# → Feature Extraction → AI Engine → Maintenance Decision

# %%

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["axes.grid"] = True

# %% [markdown]
# ## 1. Full Order Model (FOM) of Industrial DC Motor Drive
#
# **States**
# - x1 = i_a : Armature current [A]
# - x2 = ω   : Angular speed [rad/s]
# - x3 = θ   : Shaft position [rad]
# - x4 = T_j : Junction temperature / health proxy [°C]

# %%

MOTOR_PARAMS = {
    "Ra": 0.5,      # Armature resistance [ohm]
    "La": 0.01,     # Armature inductance [H]
    "Ke": 0.1,      # Back EMF constant [V/(rad/s)]
    "Kt": 0.1,      # Torque constant 
    "J": 0.01,      # Rotor inertia 
    "B": 0.001,     # Viscous friction coefficient 
    "tau_th": 50.0, # Thermal time constant 
    "k_cu": 0.05,   # Copper loss gain for i^2 heating
    "T_amb": 25.0,  # Ambient temperature 
}


def motor_fom_dynamics(t, x, v_a, T_load, params, fault_profile=None):
    """Full Order Model dynamics for DC motor.

    x = [i_a, omega, theta, Tj]
    """
    i_a, omega, theta, Tj = x
    Ra = params["Ra"]
    La = params["La"]
    Ke = params["Ke"]
    Kt = params["Kt"]
    J = params["J"]
    B = params["B"]
    tau_th = params["tau_th"]
    k_cu = params["k_cu"]
    T_amb = params["T_amb"]

    # Apply fault modifications (if any)
    extra_friction = 0.0
    Ra_fault = 0.0
    k_cu_factor = 1.0

    if fault_profile is not None:
        # bearing degradation of increased friction
        if fault_profile.get("type") == "bearing_degradation":
            extra_friction = fault_profile.get("delta_B", 0.0)
        # winding issue / overload of higher Ra
        if fault_profile.get("type") == "winding_increase_Ra":
            Ra_fault = fault_profile.get("delta_Ra", 0.0)
        # thermal overload of stronger heating
        if fault_profile.get("type") == "thermal_overload":
            k_cu_factor = fault_profile.get("k_cu_factor", 1.5)

    Ra_eff = Ra + Ra_fault
    B_eff = B + extra_friction
    k_cu_eff = k_cu * k_cu_factor

    # Electrical dynamics
    di_dt = (v_a - Ra_eff * i_a - Ke * omega) / La

    # Mechanical dynamics
    domega_dt = (Kt * i_a - B_eff * omega - T_load) / J

    # Position
    dtheta_dt = omega

    # Thermal / health proxy
    dTj_dt = (k_cu_eff * i_a**2 - (Tj - T_amb)) / tau_th

    return [di_dt, domega_dt, dtheta_dt, dTj_dt]



def simulate_motor_fom(
    t_final=5.0,
    dt=0.001,
    v_a_profile=lambda t: 220.0,
    T_load_profile=lambda t: 1.0,
    params=None,
    fault_profile=None,
    x0=None,
):
    """Simulate the FOM over [0, t_final] with step size dt.

    Returns t, x (states), and input trajectories.
    """
    if params is None:
        params = MOTOR_PARAMS

    if x0 is None:
        x0 = [0.0, 0.0, 0.0, params["T_amb"]]  # start at ambient temp

    t_span = (0.0, t_final)
    t_eval = np.arange(0.0, t_final, dt)

    def dyn(t, x):
        v_a = v_a_profile(t)
        T_load = T_load_profile(t)
        return motor_fom_dynamics(t, x, v_a, T_load, params, fault_profile)

    sol = solve_ivp(dyn, t_span, x0, t_eval=t_eval, method="RK45")
    t = sol.t
    x = sol.y.T  # shape (N, 4)

    v_a_traj = np.array([v_a_profile(ti) for ti in t])
    T_load_traj = np.array([T_load_profile(ti) for ti in t])

    return t, x, v_a_traj, T_load_traj


# %% [markdown]
# ### FOM Simulation Example (Healthy Operation)

# %%
t, x, v_a_traj, T_load_traj = simulate_motor_fom(
    t_final=5.0,
    dt=0.001,
    v_a_profile=lambda t: 220.0 if t > 0.1 else 0.0,  # start-up transient
    T_load_profile=lambda t: 1.0,
)

i_a = x[:, 0]
omega = x[:, 1]
theta = x[:, 2]
Tj = x[:, 3]

fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
axs[0].plot(t, v_a_traj)
axs[0].set_ylabel("Armature Voltage [V]")

axs[1].plot(t, i_a)
axs[1].set_ylabel("Armature Current [A]")

axs[2].plot(t, omega)
axs[2].set_ylabel("Speed [rad/s]")

axs[3].plot(t, Tj)
axs[3].set_ylabel("Junction Temp ")
axs[3].set_xlabel("Time [s]")
plt.suptitle("FOM DC Motor Response - Healthy Operation")
plt.tight_layout()
plt.show()

# %% [markdown]
# The FOM captures:
# - Start-up current spike
# - Speed ramp-up and steady-state
# - Position integration
# - Slow thermal rise due to copper losses
#
# This is representative of real industrial DC motor drive dynamics at the level
# needed for **control and monitoring**.

# %% [markdown]
# ## 2. Reduced Order Model (ROM)
#
# For **edge deployment**, we need a simpler model retaining **dominant dynamics**:
#
# - We keep only **electrical current** and **speed** dynamics.
# - Shaft position and detailed thermal state are omitted in the ROM and instead
#   captured indirectly via sensors and AI.
# This is effectively a **control-oriented, second-order model**, much lighter
# for real-time edge execution. We will quantitatively compare
# **FOM vs ROM accuracy & computation time**.

# %%
def motor_rom_dynamics(t, x, v_a, T_load, params, fault_profile=None):
    """Reduced Order Model dynamics.

    x = [i_a, omega]
    """
    i_a, omega = x
    Ra = params["Ra"]
    La = params["La"]
    Ke = params["Ke"]
    Kt = params["Kt"]
    J = params["J"]
    B = params["B"]

    extra_friction = 0.0
    Ra_fault = 0.0

    if fault_profile is not None:
        if fault_profile.get("type") == "bearing_degradation":
            extra_friction = fault_profile.get("delta_B", 0.0)
        if fault_profile.get("type") == "winding_increase_Ra":
            Ra_fault = fault_profile.get("delta_Ra", 0.0)

    Ra_eff = Ra + Ra_fault
    B_eff = B + extra_friction

    di_dt = (v_a - Ra_eff * i_a - Ke * omega) / La
    domega_dt = (Kt * i_a - B_eff * omega - T_load) / J

    return [di_dt, domega_dt]



def simulate_motor_rom(
    t_final=5.0,
    dt=0.001,
    v_a_profile=lambda t: 220.0,
    T_load_profile=lambda t: 1.0,
    params=None,
    fault_profile=None,
    x0=None,
):
    if params is None:
        params = MOTOR_PARAMS
    if x0 is None:
        x0 = [0.0, 0.0]

    t_span = (0.0, t_final)
    t_eval = np.arange(0.0, t_final, dt)

    def dyn(t, x):
        v_a = v_a_profile(t)
        T_load = T_load_profile(t)
        return motor_rom_dynamics(t, x, v_a, T_load, params, fault_profile)

    sol = solve_ivp(dyn, t_span, x0, t_eval=t_eval, method="RK45")
    t = sol.t
    x = sol.y.T  # shape (N, 2)

    v_a_traj = np.array([v_a_profile(ti) for ti in t])
    T_load_traj = np.array([T_load_profile(ti) for ti in t])

    return t, x, v_a_traj, T_load_traj


# %% [markdown]
# ### FOM vs ROM Comparison (Accuracy & Speed)

# %%
def compare_fom_rom():
    t_final = 5.0
    dt = 0.001
    v_profile = lambda t: 220.0 if t > 0.1 else 0.0
    T_profile = lambda t: 1.0

    # FOM
    t0 = time.perf_counter()
    t_fom, x_fom, _, _ = simulate_motor_fom(
        t_final=t_final,
        dt=dt,
        v_a_profile=v_profile,
        T_load_profile=T_profile,
    )
    t1 = time.perf_counter()
    fom_time = t1 - t0

    # ROM
    t0 = time.perf_counter()
    t_rom, x_rom, _, _ = simulate_motor_rom(
        t_final=t_final,
        dt=dt,
        v_a_profile=v_profile,
        T_load_profile=T_profile,
    )
    t1 = time.perf_counter()
    rom_time = t1 - t0

    # Extract states
    i_fom = x_fom[:, 0]
    w_fom = x_fom[:, 1]
    i_rom = x_rom[:, 0]
    w_rom = x_rom[:, 1]

    # Compute simple error metrics
    i_err = i_fom - i_rom
    w_err = w_fom - w_rom
    i_rmse = np.sqrt(np.mean(i_err**2))
    w_rmse = np.sqrt(np.mean(w_err**2))

    print("FOM simulation time [s]:", fom_time)
    print("ROM simulation time [s]:", rom_time)
    print("Speedup factor (~FOM/ROM):", fom_time / rom_time if rom_time > 0 else np.nan)
    print("Current RMSE [A]:", i_rmse)
    print("Speed RMSE [rad/s]:", w_rmse)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t_fom, i_fom, label="FOM")
    axs[0].plot(t_rom, i_rom, "--", label="ROM")
    axs[0].set_ylabel("Current [A]")
    axs[0].legend()

    axs[1].plot(t_fom, w_fom, label="FOM")
    axs[1].plot(t_rom, w_rom, "--", label="ROM")
    axs[1].set_ylabel("Speed [rad/s]")
    axs[1].set_xlabel("Time [s]")
    axs[1].legend()
    plt.suptitle("FOM vs ROM Response Comparison")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t_fom, i_err)
    axs[0].set_ylabel("Current Error [A]")
    axs[1].plot(t_fom, w_err)
    axs[1].set_ylabel("Speed Error [rad/s]")
    axs[1].set_xlabel("Time [s]")
    plt.suptitle("FOM - ROM Error")
    plt.tight_layout()
    plt.show()

    return {
        "fom_time": fom_time,
        "rom_time": rom_time,
        "speedup": fom_time / rom_time if rom_time > 0 else np.nan,
        "i_rmse": i_rmse,
        "w_rmse": w_rmse,
    }


rom_stats = compare_fom_rom()

# %% [markdown]
# The ROM preserves the dominant current and speed dynamics with small RMSE.
# The simulation time is significantly lower than the FOM, demonstrating that
# **ROM enables real-time execution on Intel edge devices** where low-latency
# and computational efficiency are critical.

# %% [markdown]
# ## 3. Sensor Modelling & Data Generation
#
# We simulate **virtual industrial sensors**:
# - Speed sensor (e.g., encoder, tachometer)
# - Current sensor (e.g., Hall-effect)
# - Temperature / vibration health proxy
#
# Each measurement includes:
# - Additive Gaussian noise
# - Constant bias
# - Slow drift (e.g., due to aging)
#
# We also define **fault scenarios**:
# - Bearing degradation of increased friction (B)
# - Increased friction / mechanical degradation
# - Thermal overload of increased heating
#
# These scenarios generate labeled data:
# - 0 = healthy
# - 1 = degraded
# - 2 = fault

# %%
def add_sensor_effects(signal, noise_std=0.01, bias=0.0, drift_rate=0.0, dt=0.001):
    """Add noise, bias, and linear drift to a signal."""
    n = len(signal)
    noise = np.random.normal(0.0, noise_std, size=n)
    drift = drift_rate * np.arange(n) * dt
    return signal + noise + bias + drift



def simulate_scenario(label, t_final=5.0, dt=0.001):
    """Simulate one scenario with a given health label.

    label:
      0 = healthy
      1 = degraded
      2 = fault
    Returns dict with time, states, sensor signals, and label.
    """
    # Define base profiles
    v_profile = lambda t: 220.0 if t > 0.1 else 0.0
    T_profile = lambda t: 1.0

    # Define fault profile depending on label
    fault_profile = None
    if label == 0:
        fault_profile = None
    elif label == 1:
        # Degraded: slight bearing degradation
        fault_profile = {"type": "bearing_degradation", "delta_B": 0.002}
    elif label == 2:
        # Fault: strong thermal overload or strong friction
        fault_type = np.random.choice(["thermal_overload", "bearing_degradation"])
        if fault_type == "thermal_overload":
            fault_profile = {"type": "thermal_overload", "k_cu_factor": 2.0}
        else:
            fault_profile = {"type": "bearing_degradation", "delta_B": 0.005}

    t, x, v_a_traj, T_load_traj = simulate_motor_fom(
        t_final=t_final,
        dt=dt,
        v_a_profile=v_profile,
        T_load_profile=T_profile,
        fault_profile=fault_profile,
    )

    i_a = x[:, 0]
    omega = x[:, 1]
    Tj = x[:, 3]

    # Sensor models (different biases/drifts for realism)
    speed_meas = add_sensor_effects(
        omega, noise_std=0.5, bias=0.0, drift_rate=0.01, dt=dt
    )
    current_meas = add_sensor_effects(
        i_a, noise_std=0.2, bias=0.05, drift_rate=0.0, dt=dt
    )
    temp_meas = add_sensor_effects(
        Tj, noise_std=0.2, bias=0.5, drift_rate=0.005, dt=dt
    )

    return {
        "t": t,
        "i_a": i_a,
        "omega": omega,
        "Tj": Tj,
        "v_a": v_a_traj,
        "T_load": T_load_traj,
        "speed_meas": speed_meas,
        "current_meas": current_meas,
        "temp_meas": temp_meas,
        "label": label,
    }


# %% [markdown]
# ### Visualize Example Sensor Signals

# %%
example = simulate_scenario(label=2, t_final=5.0, dt=0.001)

fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
axs[0].plot(example["t"], example["speed_meas"])
axs[0].set_ylabel("Speed Meas [rad/s]")
axs[1].plot(example["t"], example["current_meas"])
axs[1].set_ylabel("Current Meas [A]")
axs[2].plot(example["t"], example["temp_meas"])
axs[2].set_ylabel("Temp Meas ")
axs[2].set_xlabel("Time [s]")
plt.suptitle("Simulated Sensor Signals (Example Fault Scenario)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Dataset Creation for AI-based Predictive Maintenance
#
# We convert time-series sensor data into **window-based features** suitable for
# machine learning:
#
# For each window:
# - RMS current
# - Speed variance
# - Temperature trend (slope)
# - Mean and standard deviation of each sensor
#
# These are typical **statistical health indicators** used in industrial
# predictive maintenance.

# %%
def extract_features_from_window(t_window, speed_w, current_w, temp_w):
    """Compute features for a single time window."""
    feat = {}
    feat["current_rms"] = np.sqrt(np.mean(current_w**2))
    feat["speed_var"] = np.var(speed_w)

    # Linear trend of temperature (simple slope using polyfit)
    if len(t_window) > 1:
        slope, _ = np.polyfit(t_window - t_window[0], temp_w, 1)
    else:
        slope = 0.0
    feat["temp_trend"] = slope

    # Additional statistics
    feat["speed_mean"] = np.mean(speed_w)
    feat["speed_std"] = np.std(speed_w)
    feat["current_mean"] = np.mean(current_w)
    feat["current_std"] = np.std(current_w)
    feat["temp_mean"] = np.mean(temp_w)
    feat["temp_std"] = np.std(temp_w)
    return feat



def generate_dataset(
    n_scenarios_per_class=20,
    t_final=5.0,
    dt=0.002,
    window_size=0.5,
    window_stride=0.25,
):
    """Generate labeled dataset for health classification."""
    all_features = []
    all_labels = []

    for label in [0, 1, 2]:
        for _ in range(n_scenarios_per_class):
            sim = simulate_scenario(label, t_final=t_final, dt=dt)
            t = sim["t"]
            speed = sim["speed_meas"]
            current = sim["current_meas"]
            temp = sim["temp_meas"]

            n = len(t)
            win_len = int(window_size / dt)
            stride_len = int(window_stride / dt)

            for start in range(0, n - win_len, stride_len):
                end = start + win_len
                t_w = t[start:end]
                speed_w = speed[start:end]
                current_w = current[start:end]
                temp_w = temp[start:end]
                feat = extract_features_from_window(t_w, speed_w, current_w, temp_w)
                all_features.append(feat)
                all_labels.append(label)

    df = pd.DataFrame(all_features)
    df["label"] = all_labels
    return df


dataset = generate_dataset(
    n_scenarios_per_class=15,  # adjust to increase/decrease dataset size
    t_final=5.0,
    dt=0.002,
)

print(dataset.head())
print("Dataset shape:", dataset.shape)
print(dataset["label"].value_counts())

# %% [markdown]
# The resulting dataset contains many windows across multiple simulated
# scenarios, with labels:
# - 0 = healthy
# - 1 = degraded
# - 2 = fault
#
# This forms the basis for our **AI-based predictive maintenance**.

# %% [markdown]
# ## 5. AI-Based Predictive Maintenance
#
# We implement:
# - **Random Forest Classifier** (primary model)
# - **Logistic Regression** (baseline)
# - **Isolation Forest** (optional anomaly detector)
#
# Tasks:
# - Fault detection
# - System health classification
# - Early failure prediction via window-level classification
#
# Outputs:
# - Class probabilities (fault likelihood)
# - Confusion matrix
# - Feature importance (Random Forest)
#
# In an edge deployment, this inference can be accelerated on Intel hardware for
# **low-latency** decisions.

# %%
# Prepare features and labels
feature_cols = [
    "current_rms",
    "speed_var",
    "temp_trend",
    "speed_mean",
    "speed_std",
    "current_mean",
    "current_std",
    "temp_mean",
    "temp_std",
]

X = dataset[feature_cols].values
y = dataset["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features for Logistic Regression (tree-based RF does not require scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight="balanced",
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
y_prob_rf = rf_clf.predict_proba(X_test)

print("=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=["Healthy", "Degraded", "Fault"])
disp_rf.plot(values_format="d")
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Logistic Regression baseline
log_reg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

print("=== Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(cm_lr, display_labels=["Healthy", "Degraded", "Fault"])
disp_lr.plot(values_format="d")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# Feature importance (Random Forest)
importances = rf_clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 4))
plt.bar(range(len(feature_cols)), importances[sorted_idx])
plt.xticks(
    range(len(feature_cols)),
    np.array(feature_cols)[sorted_idx],
    rotation=45,
    ha="right",
)
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# %% [markdown]
# The Random Forest model typically outperforms Logistic Regression, and the
# **feature importance** plot explains which indicators drive maintenance
# decisions, improving **explainability** for industrial engineers.

# %% [markdown]
# ### Optional: Anomaly Detection with Isolation Forest
#
# This can be used to flag unknown or new fault types.

# %%
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)
iso_forest.fit(X_train)
scores_if = iso_forest.decision_function(X_test)
anomaly_flags = iso_forest.predict(X_test)  # -1 = anomaly, 1 = normal

print("Isolation Forest anomaly flags (sample):", anomaly_flags[:20])

# %% [markdown]
# ## 6. Digital Twin Architecture & Maintenance Decisions
#
# The implemented pipeline represents a **Smart Digital Twin**:
#
# 1. **Virtual Physical System**: FOM DC motor model (electro-mechanical + thermal).
# 2. **Sensors**: Simulated speed, current, temperature with realistic imperfections.
# 3. **Reduced Order Model (ROM)**: Lightweight model for on-edge state prediction
#    and fast simulation.
# 4. **AI Engine**: Random Forest & Logistic Regression for health classification
#    and fault probability.
# 5. **Maintenance Decision Logic**: Maps class probabilities to actionable
#    recommendations.
#
# Edge deployment on Intel platforms:
# - ROM and AI models are compact and low-latency.
# - Models can be exported and accelerated using Intel Edge AI toolchains
#   (e.g., quantized models and efficient runtimes).
#
# Below we implement a simple decision function mapping AI outputs to
# maintenance actions.

# %%
def maintenance_decision(probabilities, class_labels=[0, 1, 2], warn_threshold=0.5):
    """Map class probabilities to maintenance recommendations.

    probabilities: array-like [p_healthy, p_degraded, p_fault]
    """
    p_healthy, p_degraded, p_fault = probabilities
    if p_fault >= warn_threshold:
        action = (
            "IMMEDIATE MAINTENANCE: Plan shutdown and inspect motor (bearing & thermal)."
        )
    elif p_degraded >= warn_threshold:
        action = (
            "SCHEDULED MAINTENANCE: Inspect during next planned downtime."
        )
    else:
        action = "NORMAL OPERATION: Continue monitoring."
    return action


# Test decision mapping on some test samples
for i in range(5):
    probs = y_prob_rf[i]
    print(f"Sample {i}: True label={y_test[i]}, probs={probs}")
    print("Decision:", maintenance_decision(probs))
    print("-----")

# %% [markdown]
# ### Simple Health Timeline Visualization
#
# We can visualize the predicted class over time for one simulated run,
# mimicking online edge monitoring.

# %%
def simulate_and_classify_timeline(label=0, t_final=10.0, dt=0.002):
    sim = simulate_scenario(label, t_final=t_final, dt=dt)
    t = sim["t"]
    speed = sim["speed_meas"]
    current = sim["current_meas"]
    temp = sim["temp_meas"]

    win_len = int(0.5 / dt)
    stride_len = int(0.25 / dt)

    predicted_labels = []
    times = []

    for start in range(0, len(t) - win_len, stride_len):
        end = start + win_len
        t_w = t[start:end]
        speed_w = speed[start:end]
        current_w = current[start:end]
        temp_w = temp[start:end]
        feat = extract_features_from_window(t_w, speed_w, current_w, temp_w)
        x = np.array([feat[c] for c in feature_cols]).reshape(1, -1)
        probs = rf_clf.predict_proba(x)[0]
        pred_label = np.argmax(probs)
        predicted_labels.append(pred_label)
        times.append(t_w.mean())

    return np.array(times), np.array(predicted_labels)


# Example health timeline for a fault scenario
times, preds = simulate_and_classify_timeline(label=2, t_final=8.0, dt=0.002)
plt.step(times, preds, where="post")
plt.yticks([0, 1, 2], ["Healthy", "Degraded", "Fault"])
plt.xlabel("Time [s]")
plt.ylabel("Predicted Health State")
plt.title("Predicted Health Timeline (Example Fault Scenario)")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 7. Results & Discussion
#
# - **Mechatronics integration**: The simulation results confirm that the
#   digital twin correctly captures electrical (armature voltage and current),
#   mechanical (speed, load torque, friction), and thermal dynamics of the DC
#   motor drive under different operating conditions.
#
# - **Reduced Order Modelling (ROM)**: The ROM closely tracks the Full Order
#   Model for current and speed while requiring significantly lower computation
#   time. RMSE values remain low across tested scenarios, validating the ROM for
#   real-time edge execution.
#
# - **AI/ML performance**: The Random Forest classifier successfully
#   distinguishes between healthy, degraded, and fault conditions. Confusion
#   matrices and classification reports provide quantitative evidence of
#   predictive maintenance capability.
#
# - **Computational efficiency**: Feature extraction and model inference are
#   lightweight, making the approach suitable for deployment on Intel-based
#   edge devices such as industrial gateways and IPCs.
#
# - **Industrial relevance**: The modeled behaviors and fault scenarios are
#   representative of industrial DC motor drives used in applications such as
#   semiconductor fabs, conveyor systems, and pumping units.
#
# This section concludes the end-to-end **Edge-Ready Smart Digital Twin**
# implementation and analysis.