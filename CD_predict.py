import numpy as np
import pandas as pd
import sys
import os

# ── Configuration ──────────────────────────────────────────────
ROCKET_NAME  = "Tarc"   # only diameter, parachute_area, parachute_cd used
LAUNCH_CSV   = "launch_data.csv"

TOLERANCE_FT = 1.0
MAX_ITER     = 50
CD_MIN       = 0.1
CD_MAX       = 2.0
# ───────────────────────────────────────────────────────────────

MOTOR_MAP = {
    "F-51": ("AeroTech_F51NT_ThrustCurve.csv",  0.082,  0.0575, 6),
    "F51":  ("AeroTech_F51NT_ThrustCurve.csv",  0.082,  0.0575, 6),
    "G38":  ("AeroTech_G38_ThrustCurve.csv",    0.128,  0.0657, 7),
    "G-38": ("AeroTech_G38_ThrustCurve.csv",    0.128,  0.0657, 7),
    "G80":  ("AeroTech_G80T-7_ThrustCurve.csv", 0.128,  0.0657, 7),
    "G-80": ("AeroTech_G80T-7_ThrustCurve.csv", 0.128,  0.0657, 7),
    "F35":  ("AeroTech_F35W_ThrustCurve.csv",   0.085,  0.050,  6),
    "F-35": ("AeroTech_F35W_ThrustCurve.csv",   0.085,  0.050,  6),
}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as sim


def load_rocket_dims(rocket_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", os.path.join("Rockets", f"{rocket_name}.py"))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg.diameter, cfg.parachute_area, cfg.parachute_cd


def resolve_motor(motor_str):
    key = motor_str.strip()
    if key in MOTOR_MAP:
        fname, m_init, m_final, delay = MOTOR_MAP[key]
        return os.path.join("Motor CSVS", fname), m_init, m_final, delay
    raise ValueError(f"No motor found for '{motor_str}'")



def get_apogee_ft(cd, diameter, parachute_area, parachute_cd,
                  mass_dry, motor_csv, motor_mass_initial, motor_mass_final,
                  ejection_delay, launch_site):

    # Patch main.py globals so its wind functions use the right weather
    sim.launch_site = launch_site

    ts = np.linspace(0, 5, 5000)
    # need a temporary rocket just to get the thrust func for burnout detection
    temp_rocket = sim.Rocket(
        mass_dry           = mass_dry,
        motor_mass_initial = motor_mass_initial,
        motor_mass_final   = motor_mass_final,
        burn_time          = 1.5,
        diameter           = diameter,
        drag_coefficient   = cd,
        parachute_area     = parachute_area,
        parachute_cd       = parachute_cd,
        thrust_csv         = motor_csv,
    )
    burnout_time = next((t for t in ts if temp_rocket.thrust_func(t) <= 1.0), 1.5)

    rocket = sim.Rocket(
        mass_dry           = mass_dry,
        motor_mass_initial = motor_mass_initial,
        motor_mass_final   = motor_mass_final,
        burn_time          = burnout_time,
        diameter           = diameter,
        drag_coefficient   = cd,
        parachute_area     = parachute_area,
        parachute_cd       = parachute_cd,
        thrust_csv         = motor_csv,
    )
    rocket.ejection_delay = ejection_delay
    parachute_deploy_time = burnout_time + ejection_delay

    trajectory, _, _ = sim.simulate(rocket, burnout_time, parachute_deploy_time,
                                    launch_angle_deg=90)
    _, _, altitudes, _ = zip(*trajectory)
    return max(altitudes) * 3.28084


def binary_search_cd(target_ft, diameter, parachute_area, parachute_cd,
                     mass_dry, motor_csv, motor_mass_initial, motor_mass_final,
                     ejection_delay, launch_site):
    low, high = CD_MIN, CD_MAX

    for i in range(MAX_ITER):
        mid    = (low + high) / 2
        apogee = get_apogee_ft(mid, diameter, parachute_area, parachute_cd,
                               mass_dry, motor_csv, motor_mass_initial, motor_mass_final,
                               ejection_delay, launch_site)
        error  = apogee - target_ft
        print(f"    iter {i+1:2d}: CD={mid:.4f}  →  {apogee:.1f} ft  (err={error:+.1f} ft)")

        if abs(error) <= TOLERANCE_FT:
            print(f"    Converged: CD = {mid:.4f}")
            return mid

        if error > 0:
            low = mid
            print(f"             → too high, increasing CD  [{low:.4f}–{high:.4f}]")
        else:
            high = mid
            print(f"             → too low,  decreasing CD  [{low:.4f}–{high:.4f}]")

    mid = (low + high) / 2
    print(f"    Max iterations reached. Best CD = {mid:.4f}")
    return mid


def main():
    diameter, parachute_area, parachute_cd = load_rocket_dims(ROCKET_NAME)

    df = pd.read_csv(LAUNCH_CSV)
    col_map = {c: c.split('(')[0].strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    cds = []

    for idx, row in df.iterrows():
        alt_str = str(row.get("altitude", "")).strip()
        if not alt_str or alt_str.lower() == "nan":
            print(f"\nLaunch {idx+1} ({row['date']}): no altitude recorded, skipping.")
            continue

        target_ft  = float(alt_str)
        motor_csv, motor_mass_initial, motor_mass_final, ejection_delay = resolve_motor(str(row["motor"]).strip())
        mass_dry   = float(row["final_mass"]) / 1000 - motor_mass_initial

        launch_site = sim.make_launch_site(
            elevation_ft       = 0.0,
            temperature_f      = float(row["temp"]),
            pressure_inHg      = float(row["pressure"]),
            wind_speed_mph     = float(row["wind_speed"]),
            wind_direction_deg = 0.0,
            gust_speed_mph     = float(row["wind_speed"]),
            gust_frequency     = 0.1,
        )

        label = f"{row['date']} #{idx+1}"
        print(f"\nLaunch {idx+1} | {label} | target={target_ft:.0f} ft | "
              f"mass_dry={mass_dry*1000:.0f} g | motor={row['motor'].strip()}")

        cd = binary_search_cd(target_ft, diameter, parachute_area, parachute_cd,
                              mass_dry, motor_csv, motor_mass_initial, motor_mass_final,
                              ejection_delay, launch_site)
        cds.append((label, cd))
        print(f"  → CD for this launch: {cd:.4f}")

    print("\n" + "="*55)
    print("  CD RESULTS PER LAUNCH")
    print("="*55)
    for label, cd in cds:
        print(f"  {label:<20}  CD = {cd:.4f}")
    
    cd_vals = [cd for _, cd in cds]
    avg_cd  = np.mean(cd_vals)
    
    # Remove outliers beyond 1.5 IQR
    q1, q3 = np.percentile(cd_vals, 25), np.percentile(cd_vals, 75)
    iqr     = q3 - q1
    trimmed = [cd for cd in cd_vals if q1 - 1.5*iqr <= cd <= q3 + 1.5*iqr]
    avg_trimmed = np.mean(trimmed)

    print("-"*55)
    print(f"  Raw average     ({len(cd_vals)} launches):  CD = {avg_cd:.4f}")
    print(f"  Trimmed average ({len(trimmed)} launches):  CD = {avg_trimmed:.4f}  (outliers removed)")
    print("="*55)


if __name__ == "__main__":
    main()
