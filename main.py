import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import os

# ----------------------------
# CONSTANTS
# ----------------------------
GRAVITY = 9.81  # m/s^2
GAS_CONSTANT = 287.05  # J/(kg·K)

# ----------------------------
# TOGGLES
# ----------------------------
ENABLE_OPENROCKET_COMPARISON = False
ENABLE_LAUNCH_SITE_PRINT = True
ENABLE_TARC_SCORING = False

# ----------------------------
# LAUNCH SITE CONDITIONS (modifiable)
# ----------------------------
launch_site = {
    "elevation_m": 0.0,       # meters
    "temperature_C": 15.0,  
    "pressure_kPa": 101.3,    
    "wind_speed_mps": 4.4704,   
    "wind_direction_deg": 170 # from north, clockwise
}

# ----------------------------
# ATMOSPHERE MODELS
# ----------------------------

# ====BASIC ATMOSPHERE SETUP(less time)====

def air_density_at_altitude(alt_m):
    if alt_m < 11000:
        temp = 288.15 - 0.0065 * alt_m
        pressure = 101325 * (temp / 288.15)**5.2561
        return pressure / (GAS_CONSTANT * temp)
    else:
        return 1.225 * np.exp(-alt_m / 8400)


# ====ADVANCED ATMOSPHERE SETUP(takes a bit of time)====

# def load_atmosphere():
#     # Load from CSVS folder 
#     df = pd.read_csv("AtmosphereData/EarthAtmGram2007Nominal1000mc.csv", header=None)
#     altitudes = df.iloc[1:, 0].astype(float).values 
#     densities = df.iloc[1:, 1].astype(float).values
#     return interp1d(
#         altitudes,
#         densities,
#         bounds_error=False,
#         fill_value=(densities[0], densities[-1])
#     )

# # initialize interpolation function 
# print("Loading atmosphere data..."))
# _atm_density_func = load_atmosphere()
# print("Loaded Atmosphere Data")

# def air_density_at_altitude(alt_m):
#     """Return air density [kg/m^3] at altitude (meters) from CSV data."""
#     return float(_atm_density_func(max(0, alt_m))) 
# ----------------------------
# ROCKET CLASS
# ----------------------------
class Rocket:
    def __init__(self, mass_dry, motor_mass_initial, motor_mass_final, burn_time,
                 diameter, drag_coefficient, parachute_area, parachute_cd, thrust_csv):
        self.mass_dry = mass_dry
        self.motor_mass_initial = motor_mass_initial
        self.motor_mass_final = motor_mass_final
        self.burn_time = burn_time
        self.area = np.pi * (diameter / 2)**2
        self.drag_coefficient = drag_coefficient
        self.parachute_area = parachute_area
        self.parachute_cd = parachute_cd
        self.thrust_func = self.load_thrust_profile(thrust_csv)

    def load_thrust_profile(self, csv_path):
        df = pd.read_csv(csv_path, skiprows=4)
        times = df["Time (s)"].values.astype(float)
        thrusts = df["Thrust (N)"].values.astype(float)
        if times[0] > 0.0:
            times = np.insert(times, 0, 0.0)
            thrusts = np.insert(thrusts, 0, thrusts[0])
        return interp1d(times, thrusts, bounds_error=False, fill_value=0.0)

    def total_mass(self, t):
        if t >= self.burn_time:
            return self.mass_dry + self.motor_mass_final
        burn_fraction = max(0, t / self.burn_time)
        propellant_burned = (self.motor_mass_initial - self.motor_mass_final) * burn_fraction
        return self.mass_dry + self.motor_mass_initial - propellant_burned

# ----------------------------
# RK4 STEP
# ----------------------------
def rk4_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + dt/2 * k1, *args)
    k3 = f(t + dt/2, y + dt/2 * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# ----------------------------
# FORCE MODEL
# ----------------------------
def compute_force(t, y, rocket, parachute, parachute_fully_inflated,
                  inflation_start_time, inflation_duration, current_time, launch_angle):
    x, altitude, vx, vy = y[0,0], y[1,0], y[2,0], y[3,0]
    velocity_mag = np.sqrt(vx**2 + vy**2)
    rho = air_density_at_altitude(altitude)
    mass = rocket.total_mass(t)

    # parachute modeling
    if parachute:
        if parachute_fully_inflated:
            cd, area = rocket.parachute_cd, rocket.parachute_area
        else:
            inflation_progress = min((current_time - inflation_start_time)/inflation_duration, 1.0)
            cd, area = rocket.parachute_cd * inflation_progress, rocket.parachute_area
    else:
        cd, area = rocket.drag_coefficient, rocket.area

    # drag forces
    if velocity_mag > 0:
        drag_force = 0.5 * rho * cd * area * velocity_mag**2
        drag_x = -drag_force * (vx / velocity_mag)
        drag_y = -drag_force * (vy / velocity_mag)
    else:
        drag_x = drag_y = 0

    # thrust forces
    thrust = rocket.thrust_func(t)
    thrust_x = thrust * np.cos(launch_angle)
    thrust_y = thrust * np.sin(launch_angle)

    ax = (thrust_x + drag_x)/mass
    ay = (thrust_y + drag_y)/mass - GRAVITY

    return np.array([[vx], [vy], [ax], [ay]])

# ----------------------------
# SIMULATION LOOP
# ----------------------------
def simulate(rocket, burnout_time, parachute_deploy_time, launch_angle_deg=88, dt=0.001):
    launch_angle = np.radians(launch_angle_deg)
    y = np.array([[0.0],[0.0],[0.0],[0.0]])
    t = 0.0
    trajectory = [(t, y[0,0], y[1,0], 0.0)]

    parachute = False
    parachute_time = None
    parachute_fully_inflated = False
    inflation_duration = 1.0
    inflation_start_time = None
    landed_time = None
    post_landing_duration = 10.0

    latch = False

    while True:
        
        if y[1,0] >= 200 and not latch:
            rocket.drag_coefficient = 0.75  # higher drag at high speeds
            rocket.area = 50
            latch = True
        
        if not parachute and t >= parachute_deploy_time:
            parachute = True
            parachute_time = t
            inflation_start_time = t

        if parachute and not parachute_fully_inflated and t - inflation_start_time >= inflation_duration:
            parachute_fully_inflated = True

        y = rk4_step(compute_force, t, y, dt, rocket, parachute, parachute_fully_inflated,
                     inflation_start_time, inflation_duration, t, launch_angle)
        t += dt
        trajectory.append((t, y[0,0], y[1,0], y[3,0]))

        if y[1,0] <= 0.0 and t > 1.0 and landed_time is None:
            landed_time = t

        if landed_time and t >= landed_time + post_landing_duration:
            break

        if y[1,0] < 0:
            y[1,0] = 0
            y[3,0] = 0

    return trajectory, parachute_time, landed_time

# ----------------------------
# PLOTTING & OPENROCKET COMPARISON
# ----------------------------
def plot_trajectory(trajectory, parachute_time, landed_time, openrocket_csv_path=None):
    times, x_positions, altitudes, velocities = zip(*trajectory)
    positions_ft = [alt*3.28084 for alt in altitudes]
    x_positions_ft = [x*3.28084 for x in x_positions]

    accelerations = [(v2-v1)/(t2-t1) for (t1,_,_,v1),(t2,_,_,v2) in zip(trajectory[:-1], trajectory[1:])]
    accelerations.append(accelerations[-1])

    fig, axs = plt.subplots(3,2,figsize=(14,12))
    axs = axs.flatten()

    # altitude vs time graph
    max_altitude = max(positions_ft)
    apogee_time = times[positions_ft.index(max_altitude)]
    axs[0].plot(times, positions_ft, label="Altitude (ft)", color='tab:blue')
    axs[0].axvline(apogee_time,color='red',linestyle='--',label='Apogee')
    if parachute_time:
        axs[0].axvline(parachute_time,color='blue',linestyle='--',label='Parachute')
    if landed_time:
        axs[0].axvline(landed_time,color='green',linestyle='--',label='Landing')
    axs[0].set_title("Altitude vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Altitude (ft)")
    axs[0].legend()
    axs[0].grid(True)

    # velocity vs time graph
    axs[1].plot(times, velocities,label="Velocity (m/s)",color='tab:orange')
    axs[1].set_title("Velocity vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()
    axs[1].grid(True)

    # acceleration vs time
    axs[2].plot(times, accelerations,label="Acceleration (m/s²)",color='tab:green')
    axs[2].set_title("Acceleration vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Acceleration (m/s²)")
    axs[2].legend()
    axs[2].grid(True)

    # --- top down flight  path ---
    wind_speed = launch_site["wind_speed_mps"]
    wind_dir_deg = launch_site["wind_direction_deg"]
    wind_dir_rad = np.radians(wind_dir_deg)

    # wind vs movement
    wind_vx = -wind_speed * np.sin(wind_dir_rad)
    wind_vy = -wind_speed * np.cos(wind_dir_rad)

    # ground drift 
    drift_x = 0.0
    drift_y = 0.0
    dt = times[1] - times[0]
    descent_started = False
    for i in range(1, len(times)):
        if not descent_started and altitudes[i] < altitudes[i-1]:
            descent_started = True
        if descent_started:
            drift_x += wind_vx * dt
            drift_y += wind_vy * dt

    total_drift_ft = np.sqrt(drift_x**2 + drift_y**2) * 3.28084

    # quadrant prediction
    if wind_vx >= 0 and wind_vy >= 0:
        quadrant = "Quadrant I (Northeast)"
    elif wind_vx < 0 and wind_vy >= 0:
        quadrant = "Quadrant II (Northwest)"
    elif wind_vx < 0 and wind_vy < 0:
        quadrant = "Quadrant III (Southwest)"
    else:
        quadrant = "Quadrant IV (Southeast)"

    # top down plot
    axs[3].axhline(0, color='gray', linewidth=1)
    axs[3].axvline(0, color='gray', linewidth=1)
    axs[3].set_aspect('equal', 'box')
    axs[3].set_title("Top-Down Flight Path (Wind Drift Map)")

    axs[3].scatter(0, 0, color='green', s=80, label="Launch Site")
    axs[3].arrow(0, 0, drift_x * 3.28084, drift_y * 3.28084,
                 head_width=10, head_length=10, color='red', label="Drift Vector")

    axs[3].set_xlabel("East-West Position (ft)")
    axs[3].set_ylabel("North-South Position (ft)")
    axs[3].legend()
    axs[3].grid(True)

    axs[3].text( 100,  100, "NE", fontsize=12, color='gray')
    axs[3].text(-100,  100, "NW", fontsize=12, color='gray')
    axs[3].text(-100, -100, "SW", fontsize=12, color='gray')
    axs[3].text( 100, -100, "SE", fontsize=12, color='gray')

    print(f"\nPredicted Drift Distance: {total_drift_ft:.1f} ft")
    print(f"Predicted Drift Quadrant: {quadrant}")


    # openRocket comparison if selected 
    if openrocket_csv_path:
        try:
            df = pd.read_csv(openrocket_csv_path, comment='#', header=None)
            df.columns = [f"c{i}" for i in range(df.shape[1])]
            axs[4].plot(df["c0"], df["c1"], '--', color='tab:purple', label="OpenRocket Altitude")
            axs[4].plot(times, positions_ft, color='tab:blue', label="Python Sim Altitude")
            axs[4].set_title("Altitude Comparison with OpenRocket")
            axs[4].set_xlabel("Time (s)")
            axs[4].set_ylabel("Altitude (ft)")
            axs[4].legend()
            axs[4].grid(True)

            axs[5].plot(df["c0"], df["c3"], '--', color='tab:red', label="OpenRocket Velocity")
            axs[5].plot(times, velocities, color='tab:orange', label="Python Sim Velocity")
            axs[5].set_title("Velocity Comparison with OpenRocket")
            axs[5].set_xlabel("Time (s)")
            axs[5].set_ylabel("Velocity (m/s)")
            axs[5].legend()
            axs[5].grid(True)

        except Exception as e:
            axs[4].text(0.5,0.5,f"OpenRocket comparison failed:\n{e}",ha='center',va='center')
            axs[4].axis('off')
            axs[5].axis('off')
    else:
        axs[4].axis('off')
        axs[5].axis('off')

    plt.tight_layout()
    plt.show()


# ----------------------------
# PRINT FUNCTIONS
# ----------------------------
def print_rocket_info(rocket, launch_angle):
    print("\n" + "-"*60)
    print("                    ROCKET SPECIFICATIONS")
    print("-"*60)
    print(f"Rocket Dry Mass:        {rocket.mass_dry*1000:.1f} g")
    print(f"Motor Mass (loaded):    {rocket.motor_mass_initial*1000:.1f} g")
    print(f"Motor Mass (empty):     {rocket.motor_mass_final*1000:.1f} g")
    print(f"Total Launch Mass:      {(rocket.mass_dry + rocket.motor_mass_initial)*1000:.1f} g")
    print(f"Propellant Mass:        {(rocket.motor_mass_initial - rocket.motor_mass_final)*1000:.1f} g")
    diameter = np.sqrt(rocket.area/np.pi)*2
    print(f"Diameter:               {diameter*100:.1f} cm")
    print(f"Cross-sectional Area:   {rocket.area*10000:.1f} cm^2")
    print(f"Drag Coefficient:       {rocket.drag_coefficient:.3f}")
    parachute_diameter = np.sqrt(rocket.parachute_area/np.pi)*2
    print(f"Parachute Diameter:     {parachute_diameter*100:.1f} cm")
    print(f"Parachute Area:         {rocket.parachute_area:.3f} m^2")
    print(f"Parachute Cd:           {rocket.parachute_cd:.1f}")

def print_launch_site_info(launch_angle):
    if ENABLE_LAUNCH_SITE_PRINT:
        print("\n" + "-"*60)
        print("                    LAUNCH SITE CONDITIONS")
        print("-"*60)
        print(f"Launch Angle:           {launch_angle} deg")
        print(f"Launch Site Elevation:  {launch_site['elevation_m']} m")
        print(f"Air Density:            {air_density_at_altitude(launch_site['elevation_m']):.3f} kg/m^3")
        print(f"Temperature:            {launch_site['temperature_C']} C")
        print(f"Pressure:               {launch_site['pressure_kPa']} kPa")
        print(f"Wind Speed:             {launch_site['wind_speed_mps']} m/s")

def print_flight_stats(trajectory, burnout_time, parachute_time):
    times, x_positions, altitudes, velocities = zip(*trajectory)
    max_alt = max(altitudes)
    max_vel = max(velocities)
    final_x = x_positions[-1]
    total_time = times[-1]
    print("\n" + "-"*60)
    print("                    FLIGHT PERFORMANCE SUMMARY")
    print("-"*60)
    print(f"Max Altitude:        {max_alt*3.28084:.0f} ft")
    print(f"Max Velocity:        {max_vel:.1f} m/s")
    print(f"Motor Burnout Time:  {burnout_time:.2f} s")
    print(f"Parachute Deploy:    {parachute_time:.2f} s")
    print(f"Total Flight Time:   {total_time:.1f} s")
    print(f"Horizontal Distance: {final_x*3.28084:.0f} ft")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print("\n" + "-"*60)
    print("                    ROCKET FLIGHT SIMULATOR")
    print("-"*60)

    # rocket info
    rocket = Rocket(
        mass_dry=0.490,
        motor_mass_initial=0.082,
        motor_mass_final=0.0555,
        burn_time=0.986,
        diameter=0.068,
        drag_coefficient=0.6,
        parachute_area=0.1294,
        parachute_cd=0.8,
        thrust_csv=os.path.join("Motor CSVS", "AeroTech_F51NT_ThrustCurve.csv")
    )

    # launch parameters
    launch_angle = 88  # degrees, slightly off vertical for horizontal motion
    print_rocket_info(rocket, launch_angle)
    print_launch_site_info(launch_angle)

    # motor burnout and parachute deployment
    ts = np.linspace(0, 5, 5000)
    burnout_time = next((t for t in ts if rocket.thrust_func(t) <= 1.0), 1.5)
    parachute_deploy_time = burnout_time + 3x

    # simulate the flight
    trajectory, parachute_time, landed_time = simulate(
        rocket, burnout_time, parachute_deploy_time, launch_angle_deg=launch_angle
    )

    # plot trajectory and OpenRocket comparison
    openrocket_csv_path = "OpenRocket Sims/Sim1FullTARC.csv" if ENABLE_OPENROCKET_COMPARISON else None
    plot_trajectory(trajectory, parachute_time, landed_time, openrocket_csv_path)
    
    # flight stats
    print_flight_stats(trajectory, burnout_time, parachute_time)

    # TARC compeition score calculator
    if ENABLE_TARC_SCORING:
        def compute_tarc_scores(trajectory):
            times, x_positions, altitudes, _ = zip(*trajectory)
            apogee_ft = max(altitudes)*3.28084
            total_flight_time = times[-1]
            landing_x = x_positions[-1]
            
            # altitude score
            altitude_score = abs(apogee_ft - 750)
            
            # time score
            if 36 <= total_flight_time <= 39:
                Time_score = 0
            elif total_flight_time < 36:
                Time_score = 4*abs(36 - total_flight_time)
            elif total_flight_time > 39:
                Time_score = 4*abs(total_flight_time - 39) 
            
            # total score
            total_score = altitude_score + Time_score
            
            print("\nTARC Scores:")
            print(f"Altitude Score: {altitude_score:.1f}")
            print(f"Time Score: {Time_score:.1f}")
            print(f"Total Score: {total_score:.1f}")
        
        compute_tarc_scores(trajectory)

