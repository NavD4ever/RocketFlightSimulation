# Rocket Flight Simulator

An advanced rocket flight simulation system with wind drift modeling, TARC competition scoring, and OpenRocket comparison capabilities.

## Project Structure

```
rocket_sim/
├── main.py               # Main simulation script with all features 
├── README.md              # This documentation
├── archive/               # Empty archive directory
├── CSVS/                  # Atmospheric data
│   └── EarthAtmGram2007Nominal1000mc.csv
├── Motor CSVS/            # Thrust curve data files(add your own if needed)
│   ├── AeroTech_F51NT_ThrustCurve.csv
│   ├── AeroTech_G38_ThrustCurve.csv
│   └── AeroTech_G80T-7_ThrustCurve.csv
└── OpenRocket Sims/       # OpenRocket comparison data
    ├── Sim1Full.csv
    └── Sim1FullTARC.csv
```

## Usage

**Run simulation**: `python main2.py`

All configuration is done directly in the main2.py file by modifying the rocket parameters and launch site conditions.

## Configuration

All configuration is done in main2.py by editing these sections:
- **Rocket specifications**: mass_dry, diameter, drag_coefficient, parachute_area
- **Motor parameters**: thrust_csv file path, burn_time, motor masses
- **Launch site conditions**: elevation, temperature, pressure, wind speed/direction
- **Feature toggles**: ENABLE_OPENROCKET_COMPARISON, ENABLE_TARC_SCORING, etc.

## Features

- **6-DOF Flight Simulation**: RK4 integration with realistic physics
- **Wind Drift Modeling**: Configurable wind speed and direction with real-time drift calculation
- **Launch Site Conditions**: Customizable elevation, temperature, pressure settings
- **Top-Down Flight Path**: Visual wind drift prediction with quadrant mapping
- **TARC Competition Scoring**: Automated scoring for Team America Rocketry Challenge
- **Atmospheric Models**: Basic standard atmosphere or advanced CSV-based data
- **Enhanced Plotting**: 6-panel comprehensive flight visualization
- **OpenRocket Comparison**: Side-by-side comparison with OpenRocket simulation data
- **Feature Toggles**: Enable/disable specific analysis components
- **Parachute Modeling**: Realistic deployment and inflation physics

## Wind Drift Analysis

The enhanced simulator includes wind modeling capabilities:
- Configurable wind speed (m/s) and direction (degrees from north)
- Real-time drift calculation during descent phase
- Top-down flight path visualization with quadrant prediction
- Landing zone estimation for recovery planning

## TARC Scoring

Built-in Team America Rocketry Challenge scoring:
- Altitude accuracy scoring (target: 800ft)
- Flight time scoring (target: 36-39 seconds)
- Automatic penalty calculation
- Competition-ready performance metrics

## Atmospheric Models

**Basic Model** (default): Standard atmosphere with altitude-dependent density [QUICK]
**Advanced Model** (optional): Highly accurate atmospheric data from CSV files [LENGTHY]
- Uses CSVS/EarthAtmGram2007Nominal1000mc.csv for more accurate density profiles
- Uncomment the advanced atmosphere section in main2.py to enable

## Motor Data

Supported motor thrust curves:
- AeroTech F51NT (default)
- AeroTech G38
- AeroTech G80T-7

Add custom motor data by placing CSV files in the Motor CSVS/ directory.

## Openrocket Comparison

Export a CSV from openrocket sims with all data selected
- Place the CSV in OpenRocket Sims/ directory
- Toggle the comparison in main2.py

## Configuration

There are a few ways to configure the simulation to your needs:
- At the top, from line 10-29, you may change the constants, launch site parameters, and toggles.
- From line 31-62, you may change which atmosphere model is used.
- From line 384-395, you may change the information of the rocket being used.
- On line 398, you may change the angle of the launch rail.