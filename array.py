import numpy as np
import sys
import os

# ── config ──────────────────────────────────────────────
ROCKET_NAME  = "Tarc"
WEATHER_NAME = "3.29"   # base weather file (elev)
TARGET_ALT_FT = 750.0
LAUNCH_ANGLE  = 90

MASS_MIN     = 0.55
MASS_MAX     = 0.65
TOLERANCE_FT = 0.5
MAX_ITER     = 100

# entry: (title, temp_f, pressure_inHg, wind_speed_mph, wind_dir_deg, gust_speed_mph)
CONDITIONS = [                            
    ("4:00 PM",  59, 30.29, 16, 295, 26)
]
# ───────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as sim


def get_apogee_ft(mass_dry, launch_site):
    rocket = sim.load_rocket(ROCKET_NAME)
    rocket.mass_dry = mass_dry
    sim.launch_site = launch_site

    ts = np.linspace(0, 5, 5000)
    burnout_time = next((t for t in ts if rocket.thrust_func(t) <= 1.0), 1.5)
    rocket.burn_time = burnout_time
    parachute_deploy_time = burnout_time + rocket.ejection_delay

    trajectory, _, _ = sim.simulate(rocket, burnout_time, parachute_deploy_time,
                                    launch_angle_deg=LAUNCH_ANGLE)
    _, _, altitudes, _ = zip(*trajectory)
    return max(altitudes) * 3.28084


def binary_search_mass(launch_site, title):
    low, high = MASS_MIN, MASS_MAX

    print(f"  checking bounds...")
    apogee_low  = get_apogee_ft(low,  launch_site)
    print(f"  bound low:  mass={low:.4f} kg → {apogee_low:.1f} ft")
    apogee_high = get_apogee_ft(high, launch_site)
    print(f"  bound high: mass={high:.4f} kg → {apogee_high:.1f} ft")

    if not (apogee_high <= TARGET_ALT_FT <= apogee_low):
        best_mass = low if abs(apogee_low - TARGET_ALT_FT) < abs(apogee_high - TARGET_ALT_FT) else high
        best_apogee = apogee_low if best_mass == low else apogee_high
        print(f"  Target {TARGET_ALT_FT} ft not bracketed [{apogee_high:.1f}–{apogee_low:.1f} ft]. Returning closest bound.")
        print(f"  Best available: {best_mass*1000:.1f} g → {best_apogee:.1f} ft")
        return best_mass

    for i in range(MAX_ITER):
        mid    = (low + high) / 2
        apogee = get_apogee_ft(mid, launch_site)
        error  = apogee - TARGET_ALT_FT
        print(f"  iter {i+1:2d}: mass={mid:.4f} kg → {apogee:.1f} ft  (err={error:+.1f} ft)")

        if abs(error) <= TOLERANCE_FT:
            print(f"  Converged: {mid*1000:.1f} g")
            return mid

        if error > 0:
            low = mid
            print(f"            → too high, increasing mass [{low:.4f}–{high:.4f} kg]")
        else:
            high = mid
            print(f"            → too low,  decreasing mass [{low:.4f}–{high:.4f} kg]")

    mid = (low + high) / 2
    print(f"  Max iterations. Best: {mid*1000:.1f} g")
    return mid


def main():
    base_weather = sim.load_weather(WEATHER_NAME)
    results = []

    for title, temp_f, pressure_inHg, wind_mph, wind_dir, gust_mph in CONDITIONS:
        print(f"\n{'='*55}")
        print(f"  {title}  |  {temp_f}°F  {pressure_inHg} inHg  {wind_mph} mph  {wind_dir}°  gusts {gust_mph} mph")
        print(f"{'='*55}")

        launch_site = sim.make_launch_site(
            elevation_ft       = base_weather["elevation_m"] / 0.3048,
            temperature_f      = temp_f,
            pressure_inHg      = pressure_inHg,
            wind_speed_mph     = wind_mph,
            wind_direction_deg = wind_dir,
            gust_speed_mph     = gust_mph,
            gust_frequency     = base_weather["gust_frequency"],
        )

        mass = binary_search_mass(launch_site, title)
        apogee = get_apogee_ft(mass, launch_site) if mass else None
        results.append((title, temp_f, pressure_inHg, wind_mph, mass, apogee))

    # Summary table
    print(f"\n{'='*65}")
    print(f"  {'CONDITION':<12} {'TEMP':>6} {'PRESSURE':>10} {'WIND':>6}  {'PRED MASS':>10}  {'APOGEE':>9}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*6}  {'-'*10}  {'-'*9}")
    for title, temp_f, pressure_inHg, wind_mph, mass, apogee in results:
        mass_str   = f"{mass*1000:.1f} g"  if mass   else "N/A"
        apogee_str = f"{apogee:.1f} ft"    if apogee else "N/A"
        print(f"  {title:<12} {temp_f:>5}°F  {pressure_inHg:>8} inHg  {wind_mph:>4} mph  {mass_str:>10}  {apogee_str:>9}")
    print(f"{'='*65}")


if __name__ == "__main__":
    print(f"Array prediction: rocket='{ROCKET_NAME}'  target={TARGET_ALT_FT} ft\n")
    main()
