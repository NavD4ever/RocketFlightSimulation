import os

# TARC Rocket 
mass_dry = 0.498  # kg
motor_mass_initial = 0.085  # kg
motor_mass_final = 0.055  # kg
burn_time = 0.986
diameter = 0.0668  # m
drag_coefficient = 0.4774
parachute_area = 0.4572  # m^2
parachute_cd = 0.8
ejection_delay = 6  # seconds
thrust_csv = os.path.join("Motor CSVS", "AeroTech_F35W_ThrustCurve.csv")