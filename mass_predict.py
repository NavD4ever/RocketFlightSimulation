import numpy as np
import sys
import os

# ── Configuration ──────────────────────────────────────────────
ROCKET_NAME     = "Tarc"
WEATHER_NAME    = "3.29"
TARGET_ALT_FT   = 750.0   # target apogee in feet
LAUNCH_ANGLE    = 90      # degrees

MASS_MIN        = 0.55    # kg  (lower bound for search)
MASS_MAX        = 0.65    # kg  (upper bound for search)
TOLERANCE_FT    = 0.5    # stop when within this many feet
MAX_ITER        = 100
# ───────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
import main as sim

def get_apogee_ft(mass_dry):
    """Run one simulation with the given dry mass and return apogee in feet."""
    rocket = sim.load_rocket(ROCKET_NAME)
    rocket.mass_dry = mass_dry

    # Patch the module-level launch_site so get_wind_at_time uses the right weather
    sim.launch_site = sim.load_weather(WEATHER_NAME)

    ts = np.linspace(0, 5, 5000)
    burnout_time = next((t for t in ts if rocket.thrust_func(t) <= 1.0), 1.5)
    parachute_deploy_time = burnout_time + rocket.ejection_delay

    trajectory, _, _ = sim.simulate(rocket, burnout_time, parachute_deploy_time,
                                    launch_angle_deg=LAUNCH_ANGLE)
    _, _, altitudes, _ = zip(*trajectory)
    return max(altitudes) * 3.28084


def binary_search_mass():
    low, high = MASS_MIN, MASS_MAX

    apogee_low  = get_apogee_ft(low)
    apogee_high = get_apogee_ft(high)
    print(f"  mass={low:.4f} kg → {apogee_low:.1f} ft")
    print(f"  mass={high:.4f} kg → {apogee_high:.1f} ft")

    if not (apogee_high <= TARGET_ALT_FT <= apogee_low):
        print(f"\nTarget {TARGET_ALT_FT} ft is not bracketed by "
              f"[{apogee_high:.1f}, {apogee_low:.1f}] ft. "
              f"Adjust MASS_MIN / MASS_MAX.")
        return None

    for i in range(MAX_ITER):
        mid = (low + high) / 2
        apogee = get_apogee_ft(mid)
        error  = apogee - TARGET_ALT_FT
        print(f"  iter {i+1:2d}: mass={mid:.4f} kg → {apogee:.1f} ft  (err={error:+.1f} ft)")

        if abs(error) <= TOLERANCE_FT:
            print(f"\nConverged: mass_dry = {mid*1000:.1f} g  →  apogee = {apogee:.1f} ft")
            return mid
        # heavier rocket → lower apogee, so flip signs accordingly
        if error > 0:   # too high → need more mass
            low = mid
            print(f"          → too high, increasing mass (new range: {low:.4f}–{high:.4f} kg)")
        else:           # too low  → need less mass
            high = mid
            print(f"          → too low, decreasing mass (new range: {low:.4f}–{high:.4f} kg)")

    mid = (low + high) / 2
    print(f"\nMax iterations reached. Best estimate: mass_dry = {mid*1000:.1f} g")
    return mid


if __name__ == "__main__":
    print(f"Binary search: rocket='{ROCKET_NAME}'  weather='{WEATHER_NAME}'  "
          f"target={TARGET_ALT_FT} ft\n")
    binary_search_mass()
