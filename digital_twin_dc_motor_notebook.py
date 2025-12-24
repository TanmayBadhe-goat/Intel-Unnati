# %% [markdown]
# Edge-Ready Smart Digital Twin for Predictive Maintenance of Industrial Motor Drive Systems using Reduced-Order Modelling and AI
#
# Jupyter-style script for Intel Unnati Problem Statement 5.
# Copy cell-by-cell into a Jupyter Notebook or run cells directly in VS Code.

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

# Reproducibility and plotting defaults
np.random.seed(42)
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["axes.grid"] = True

# %% [markdown]
# 1. Full Order Model (FOM) of Industrial DC Motor Drive

# %%
# Motor and environment parameters (approximate but physically reasonable)
MOTOR_PARAMS = {
    "Ra": 0.5,      # Armature resistance [ohm]
    "La": 0.01,     # Armature inductance [H]
    "Ke": 0.1,      # Back EMF constant [V/(rad/s)]
    "Kt": 0.1,      # Torque constant [Nm/A]
    "J": 0.01,      # Rotor inertia [kg*m^2]
    "B": 0.001,     # Viscous friction coefficient [Nm*s/rad]
    "tau_th": 50.0, # Thermal time constant [s]
    "k_cu": 0.05,   # Copper loss gain for i^2 heating
    "T_amb": 25.0,  # Ambient temperature [degC]
}


def motor_fom_dynamics(t, x, v_a, T_load, params, fault_profile=None):
    """Full Order Model (FOM) dynamics for the DC motor.

    State vector x = [i_a, omega, theta, Tj]
      i_a  : armature current [A]
      omega: angular speed [rad/s]
      theta: shaft position [rad]
      Tj   : junction / health temperature proxy [degC]
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

    # Fault modifiers (default: no fault)
    extra_friction = 0.0
    Ra_fault = 0.0
    k_cu_factor = 1.0

    if fault_profile is not None:
        fault_type = fault_profile.get("type")
        if fault_type == "bearing_degradation":
            extra_friction = fault_profile.get("delta_B", 0.0)
        if fault_type == "winding_increase_Ra":
            Ra_fault = fault_profile.get("delta_Ra", 0.0)
        if fault_type == "thermal_overload":
            k_cu_factor = fault_profile.get("k_cu_factor", 1.5)

    Ra_eff = Ra + Ra_fault
    B_eff = B + extra_friction
    k_cu_eff = k_cu * k_cu_factor

    # Electrical dynamics
    di_dt = (v_a - Ra_eff * i_a - Ke * omega) / La

    # Mechanical dynamics
    domega_dt = (Kt * i_a - B_eff * omega - T_load) / J

    # Position dynamics
    dtheta_dt = omega

    # Thermal / health proxy dynamics
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

    Returns: t, x (states), v_a_traj, T_load_traj
    """

    if params is None:
        params = MOTOR_PARAMS

    if x0 is None:
        x0 = [0.0, 0.0, 0.0, params["T_amb"]]

    t_span = (0.0, t_final)
    t_eval = np.arange(0.0, t_final, dt)

    def dyn(t, x):
        v_a = v_a_profile(t)
        T_load = T_load_profile(t)
        return motor_fom_dynamics(t, x, v_a, T_load, params, fault_profile)

    sol = solve_ivp(dyn, t_span, x0, t_eval=t_eval, method="RK45")
    t = sol.t
    x = sol.y.T

    v_a_traj = np.array([v_a_profile(ti) for ti in t])
    T_load_traj = np.array([T_load_profile(ti) for ti in t])

    return t, x, v_a_traj, T_load_traj


# %% [markdown]
# Example FOM simulation (healthy operation)

# %%
if __name__ == "__main__":
    t, x, v_a_traj, T_load_traj = simulate_motor_fom(
        t_final=5.0,
        dt=0.001,
        v_a_profile=lambda tau: 220.0 if tau > 0.1 else 0.0,
        T_load_profile=lambda tau: 1.0,
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
    axs[3].set_ylabel("Junction Temp [degC]")
    axs[3].set_xlabel("Time [s]")
    plt.suptitle("FOM DC Motor Response - Healthy Operation")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# 2. Reduced Order Model (ROM)

# %%
def motor_rom_dynamics(t, x, v_a, T_load, params, fault_profile=None):
    """Reduced Order Model (ROM) with states x = [i_a, omega]."""

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
        fault_type = fault_profile.get("type")
        if fault_type == "bearing_degradation":
            extra_friction = fault_profile.get("delta_B", 0.0)
        if fault_type == "winding_increase_Ra":
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
    """Simulate the ROM over [0, t_final]."""

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
    x = sol.y.T

    v_a_traj = np.array([v_a_profile(ti) for ti in t])
    T_load_traj = np.array([T_load_profile(ti) for ti in t])

    return t, x, v_a_traj, T_load_traj


def compare_fom_rom():
    """Compare FOM and ROM accuracy and computational speed."""

    t_final = 5.0
    dt = 0.001
    v_profile = lambda tau: 220.0 if tau > 0.1 else 0.0
    T_profile = lambda tau: 1.0

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

    i_fom = x_fom[:, 0]
    w_fom = x_fom[:, 1]
    i_rom = x_rom[:, 0]
    w_rom = x_rom[:, 1]

    i_err = i_fom - i_rom
    w_err = w_fom - w_rom
    i_rmse = np.sqrt(np.mean(i_err**2))
    w_rmse = np.sqrt(np.mean(w_err**2))

    print("FOM simulation time [s]:", fom_time)
    print("ROM simulation time [s]:", rom_time)
    if rom_time > 0:
        print("Speedup factor (FOM/ROM):", fom_time / rom_time)
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
        "i_rmse": i_rmse,
        "w_rmse": w_rmse,
    }


if __name__ == "__main__":
    rom_stats = compare_fom_rom()

# %% [markdown]
# 3. Sensor Modelling and Fault Scenarios

# %%
def add_sensor_effects(signal, noise_std=0.01, bias=0.0, drift_rate=0.0, dt=0.001):
    """Add Gaussian noise, constant bias, and linear drift to a signal."""

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
    """

    v_profile = lambda tau: 220.0 if tau > 0.1 else 0.0
    T_profile = lambda tau: 1.0

    fault_profile = None
    if label == 0:
        fault_profile = None
    elif label == 1:
        # Mild bearing degradation
        fault_profile = {"type": "bearing_degradation", "delta_B": 0.002}
    elif label == 2:
        # Strong fault: either thermal overload or heavy friction
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

    speed_meas = add_sensor_effects(omega, noise_std=0.5, bias=0.0, drift_rate=0.01, dt=dt)
    current_meas = add_sensor_effects(i_a, noise_std=0.2, bias=0.05, drift_rate=0.0, dt=dt)
    temp_meas = add_sensor_effects(Tj, noise_std=0.2, bias=0.5, drift_rate=0.005, dt=dt)

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


if __name__ == "__main__":
    example = simulate_scenario(label=2, t_final=5.0, dt=0.001)

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(example["t"], example["speed_meas"])
    axs[0].set_ylabel("Speed meas [rad/s]")
    axs[1].plot(example["t"], example["current_meas"])
    axs[1].set_ylabel("Current meas [A]")
    axs[2].plot(example["t"], example["temp_meas"])
    axs[2].set_ylabel("Temp meas [degC]")
    axs[2].set_xlabel("Time [s]")
    plt.suptitle("Simulated sensor signals (fault scenario)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# 4. Dataset Creation and Feature Engineering

# %%
def extract_features_from_window(t_window, speed_w, current_w, temp_w):
    """Compute statistical features for a single time window."""

    features = {}
    features["current_rms"] = np.sqrt(np.mean(current_w**2))
    features["speed_var"] = np.var(speed_w)

    if len(t_window) > 1:
        slope, _ = np.polyfit(t_window - t_window[0], temp_w, 1)
    else:
        slope = 0.0
    features["temp_trend"] = slope

    features["speed_mean"] = np.mean(speed_w)
    features["speed_std"] = np.std(speed_w)
    features["current_mean"] = np.mean(current_w)
    features["current_std"] = np.std(current_w)
    features["temp_mean"] = np.mean(temp_w)
    features["temp_std"] = np.std(temp_w)
    return features


def generate_dataset(
    n_scenarios_per_class=20,
    t_final=5.0,
    dt=0.002,
    window_size=0.5,
    window_stride=0.25,
):
    """Generate a labeled dataset for health classification.

    Returns a pandas DataFrame with feature columns and label column.
    """

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
                feats = extract_features_from_window(t_w, speed_w, current_w, temp_w)
                all_features.append(feats)
                all_labels.append(label)

    df = pd.DataFrame(all_features)
    df["label"] = all_labels
    return df


if __name__ == "__main__":
    dataset = generate_dataset(
        n_scenarios_per_class=15,
        t_final=5.0,
        dt=0.002,
    )
    print(dataset.head())
    print("Dataset shape:", dataset.shape)
    print("Label counts:\n", dataset["label"].value_counts())

# %% [markdown]
# 5. AI-Based Predictive Maintenance (Classification Models)

# %%
# Create dataset for the remaining cells when running as script
if __name__ == "__main__":
    if "dataset" not in globals():
        dataset = generate_dataset(
            n_scenarios_per_class=15,
            t_final=5.0,
            dt=0.002,
        )

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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest classifier (primary model)
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
# 6. Optional Anomaly Detection with Isolation Forest

# %%
if __name__ == "__main__":
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )
    iso_forest.fit(X_train)
    scores_if = iso_forest.decision_function(X_test)
    anomaly_flags = iso_forest.predict(X_test)  # -1 = anomaly, 1 = normal
    print("Isolation Forest anomaly flags (first 20):", anomaly_flags[:20])

# %% [markdown]
# 7. Digital Twin Architecture and Maintenance Decisions

# %%
def maintenance_decision(probabilities, warn_threshold=0.5):
    """Map class probabilities to maintenance recommendations.

    probabilities: [p_healthy, p_degraded, p_fault]
    """

    p_healthy, p_degraded, p_fault = probabilities
    if p_fault >= warn_threshold:
        return "IMMEDIATE MAINTENANCE: Plan shutdown and inspect motor (bearing and thermal)."
    if p_degraded >= warn_threshold:
        return "SCHEDULED MAINTENANCE: Inspect during next planned downtime."
    return "NORMAL OPERATION: Continue monitoring."


if __name__ == "__main__":
    print("Sample maintenance decisions from Random Forest probabilities:")
    for i in range(5):
        probs = y_prob_rf[i]
        print(f"Sample {i}, true label={y_test[i]}, probs={probs}")
        print("Decision:", maintenance_decision(probs))
        print("---")

# %% [markdown]
# 8. Health Timeline Visualisation (Digital Twin Monitoring)

# %%
def simulate_and_classify_timeline(rf_model, label=0, t_final=10.0, dt=0.002):
    """Simulate a scenario and classify health state over time using a sliding window."""

    sim = simulate_scenario(label, t_final=t_final, dt=dt)
    t = sim["t"]
    speed = sim["speed_meas"]
    current = sim["current_meas"]
    temp = sim["temp_meas"]

    win_len = int(0.5 / dt)
    stride_len = int(0.25 / dt)

    times = []
    predicted_labels = []

    for start in range(0, len(t) - win_len, stride_len):
        end = start + win_len
        t_w = t[start:end]
        speed_w = speed[start:end]
        current_w = current[start:end]
        temp_w = temp[start:end]
        feats = extract_features_from_window(t_w, speed_w, current_w, temp_w)
        x_vec = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
        probs = rf_model.predict_proba(x_vec)[0]
        pred_label = int(np.argmax(probs))
        times.append(t_w.mean())
        predicted_labels.append(pred_label)

    return np.array(times), np.array(predicted_labels)


if __name__ == "__main__":
    times, preds = simulate_and_classify_timeline(rf_clf, label=2, t_final=8.0, dt=0.002)

    plt.step(times, preds, where="post")
    plt.yticks([0, 1, 2], ["Healthy", "Degraded", "Fault"])
    plt.xlabel("Time [s]")
    plt.ylabel("Predicted health state")
    plt.title("Predicted health timeline (fault scenario)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# End of notebook-style script.
