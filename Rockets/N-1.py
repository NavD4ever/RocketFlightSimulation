import os

# N-1 Rocket Configuration
mass_dry = 0.468  # kg
motor_mass_initial = 0.128  # kg
motor_mass_final = 0.0657  # kg
burn_time = 1.71
diameter = 0.0671  # m
drag_coefficient = 0.44
parachute_area = 0.1639  # m^2
parachute_cd = 0.8
ejection_delay = 7  # seconds
thrust_csv = os.path.join("Motor CSVS", "AeroTech_G80T-7_ThrustCurve.csv")