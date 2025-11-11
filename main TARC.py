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
ENABLE_TARC_SCORING = True

# ----------------------------
# LAUNCH SITE CONDITIONS
# ----------------------------
launch_site = {
    "elevation_m": 123.129,       # meters
    "temperature_C": 15.67,  # celsius
    "pressure_kPa": 100.7112,    # kPa
    "wind_speed_mps": 8.4,   # base wind for 28mph gusts
    "wind_direction_deg": 315, # from north, clockwise
    "gust_intensity": 0.49,   # allows gusts up to 12.5 m/s (28mph)
    "gust_frequency": 0.2     # gust frequency (Hz)
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
# WIND GUST MODEL
# ----------------------------
def get_wind_at_time(t):
    base_speed = launch_site["wind_speed_mps"]
    base_dir = launch_site["wind_direction_deg"]
    gust_intensity = launch_site["gust_intensity"]
    gust_freq = launch_site["gust_frequency"]
    
    # Add sinusoidal gusts with random phase
    gust_factor = 1 + gust_intensity * np.sin(2 * np.pi * gust_freq * t + np.sin(0.3 * t))
    wind_speed = base_speed * gust_factor
    
    # Add directional variation
    dir_variation = 15 * np.sin(2 * np.pi * gust_freq * t * 0.7)
    wind_direction = base_dir + dir_variation
    
    return wind_speed, wind_direction

# ----------------------------
# FORCE MODEL
# ----------------------------
def compute_force(t, y, rocket, parachute, parachute_fully_inflated,
                  inflation_start_time, inflation_duration, current_time, launch_angle):
    x, altitude, vx, vy = y[0,0], y[1,0], y[2,0], y[3,0]
    velocity_mag = np.sqrt(vx**2 + vy**2)
    rho = air_density_at_altitude(altitude)
    mass = rocket.total_mass(t)
    
    # Get current wind conditions
    wind_speed, wind_dir_deg = get_wind_at_time(t)
    wind_dir_rad = np.radians(wind_dir_deg)
    wind_vx = -wind_speed * np.sin(wind_dir_rad)
    wind_vy = -wind_speed * np.cos(wind_dir_rad)
    
    # Relative velocity (rocket velocity relative to wind)
    rel_vx = vx - wind_vx
    rel_vy = vy - wind_vy
    rel_velocity_mag = np.sqrt(rel_vx**2 + rel_vy**2)

    # parachute modeling
    if parachute:
        if parachute_fully_inflated:
            cd, area = rocket.parachute_cd, rocket.parachute_area
        else:
            inflation_progress = min((current_time - inflation_start_time)/inflation_duration, 1.0)
            cd, area = rocket.parachute_cd * inflation_progress, rocket.parachute_area
    else:
        cd, area = rocket.drag_coefficient, rocket.area

    # drag forces (using relative velocity)
    if rel_velocity_mag > 0:
        drag_force = 0.5 * rho * cd * area * rel_velocity_mag**2
        drag_x = -drag_force * (rel_vx / rel_velocity_mag)
        drag_y = -drag_force * (rel_vy / rel_velocity_mag)
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
        
        # airbrakes
   #     if y[1,0] >= 200 and not latch:
    #        rocket.drag_coefficient = 0.75  
    #        rocket.area = 50
    #        latch = True
        
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

    # --- top down flight path with gusts ---
    # Calculate drift with time-varying wind
    drift_x = 0.0
    drift_y = 0.0
    dt = times[1] - times[0]
    descent_started = False
    
    for i in range(1, len(times)):
        if not descent_started and altitudes[i] < altitudes[i-1]:
            descent_started = True
        if descent_started:
            # Get wind at this time
            wind_speed, wind_dir_deg = get_wind_at_time(times[i])
            wind_dir_rad = np.radians(wind_dir_deg)
            wind_vx = -wind_speed * np.sin(wind_dir_rad)
            wind_vy = -wind_speed * np.cos(wind_dir_rad)
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
    print(f"Mass Ratio:             {(rocket.mass_dry + rocket.motor_mass_initial)/(rocket.mass_dry + rocket.motor_mass_final):.2f}")
    diameter = np.sqrt(rocket.area/np.pi)*2
    print(f"Diameter:               {diameter*100:.1f} cm")
    print(f"Cross-sectional Area:   {rocket.area*10000:.1f} cm^2")
    print(f"Drag Coefficient:       {rocket.drag_coefficient:.3f}")
    parachute_diameter = np.sqrt(rocket.parachute_area/np.pi)*2
    print(f"Parachute Diameter:     {parachute_diameter*100:.1f} cm")
    print(f"Parachute Area:         {rocket.parachute_area:.3f} m^2")
    print(f"Parachute Cd:           {rocket.parachute_cd:.1f}")
    print(f"Recovery System:        Single parachute deployment")

def print_launch_site_info(launch_angle):
    if ENABLE_LAUNCH_SITE_PRINT:
        print("\n" + "-"*60)
        print("                    LAUNCH SITE CONDITIONS")
        print("-"*60)
        print(f"Launch Angle:           {launch_angle} deg")
        print(f"Launch Site Elevation:  {launch_site['elevation_m']} m")
        print(f"Air Density:            {air_density_at_altitude(launch_site['elevation_m']):.3f} kg/m³")
        print(f"Atmospheric Model:      Standard atmosphere")
        print(f"Temperature:            {launch_site['temperature_C']}°C ({launch_site['temperature_C']*9/5+32:.1f}°F)")
        print(f"Pressure:               {launch_site['pressure_kPa']} kPa ({launch_site['pressure_kPa']*0.145:.1f} psi)")
        print(f"Wind Speed:             {launch_site['wind_speed_mps']:.1f} m/s ({launch_site['wind_speed_mps']*2.237:.1f} mph)")
    print(f"Wind Direction:         {launch_site['wind_direction_deg']} degrees from north")

def print_flight_stats(trajectory, burnout_time, parachute_time):
    times, x_positions, altitudes, velocities = zip(*trajectory)
    
    # Calculate accelerations
    accelerations = [(v2-v1)/(t2-t1) for (t1,_,_,v1),(t2,_,_,v2) in zip(trajectory[:-1], trajectory[1:])]
    
    # Key metrics
    max_alt = max(altitudes)
    max_alt_ft = max_alt * 3.28084
    max_vel = max(velocities)
    max_accel = max(accelerations)
    max_g_force = max_accel / 9.81
    final_x = x_positions[-1]
    total_time = times[-1]
    landing_velocity = abs(velocities[-1])
    
    # Find times for key events
    apogee_time = times[altitudes.index(max_alt)]
    max_vel_time = times[velocities.index(max_vel)]
    max_accel_time = times[accelerations.index(max_accel)]
    
    print("\n" + "-"*60)
    print("                    FLIGHT PERFORMANCE SUMMARY")
    print("-"*60)
    print(f"Maximum Altitude:       {max_alt_ft:.0f} ft ({max_alt:.0f} m) @ {apogee_time:.2f}s")
    print(f"Maximum Velocity:       {max_vel:.1f} m/s ({max_vel*2.237:.1f} mph) @ {max_vel_time:.2f}s")
    print(f"Maximum Acceleration:   {max_accel:.1f} m/s² ({max_g_force:.1f} G) @ {max_accel_time:.2f}s")
    print(f"Motor Burnout Time:     {burnout_time:.2f} s")
    print(f"Parachute Deployment:   {parachute_time:.2f} s")
    print(f"Total Flight Time:      {total_time:.1f} s")
    print(f"Horizontal Distance:    {final_x*3.28084:.0f} ft ({final_x:.0f} m)")
    print(f"Landing Velocity:       {landing_velocity:.1f} m/s ({landing_velocity*2.237:.1f} mph)")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print("\n" + "-"*60)
    print("                    ROCKET FLIGHT SIMULATOR")
    print("-"*60)

    # rocket info
    rocket = Rocket(
        mass_dry=0.523, #kg
        motor_mass_initial=0.082, #kg
        motor_mass_final=0.0575, #kg
        burn_time=0.986,
        diameter=0.0668, #m
        drag_coefficient=0.37, 
        parachute_area=0.4572,# m^2
        parachute_cd=0.8,
        thrust_csv=os.path.join("Motor CSVS", "AeroTech_F51NT_ThrustCurve.csv")
    )

    # launch parameters
    launch_angle = 90
    print_rocket_info(rocket, launch_angle)
    print_launch_site_info(launch_angle)

    # motor burnout and parachute deployment
    ts = np.linspace(0, 5, 5000)
    burnout_time = next((t for t in ts if rocket.thrust_func(t) <= 1.0), 1.5)
    parachute_deploy_time = burnout_time + 8 # integer is ejection delay

    # simulate the flight
    trajectory, parachute_time, landed_time = simulate(
        rocket, burnout_time, parachute_deploy_time, launch_angle_deg=launch_angle
    )

    # plot trajectory and OpenRocket comparison
    openrocket_csv_path = "OpenRocket Sims/SimN-1 V1.csv" if ENABLE_OPENROCKET_COMPARISON else None
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
            
            print("\n" + "-"*60)
            print("                    TARC COMPETITION SCORING")
            print("-"*60)
            print(f"Target Altitude:        750 ft")
            print(f"Actual Altitude:        {apogee_ft:.1f} ft")
            print(f"Altitude Error:         {abs(apogee_ft - 750):.1f} ft")
            print(f"Target Flight Time:     36-39 seconds")
            print(f"Actual Flight Time:     {total_flight_time:.1f} s")
            print(f"Altitude Score:         {altitude_score:.1f} points")
            print(f"Time Score:             {Time_score:.1f} points")
            print(f"Total TARC Score:       {total_score:.1f} points")
        
        compute_tarc_scores(trajectory)
        
    print("\n" + "="*60)
    print("                    SIMULATION COMPLETE")
    print("="*60)

