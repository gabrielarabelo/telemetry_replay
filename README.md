# Telemetry Replay

A panel to visualize telemetry data, including GPS, speed, accelerometer, gyroscope, linear and angular acceleration, and more. Developed by **Gabi Rabelo** (gabriela.rabelo@gmail.com).

This tool enables you to analyze and replay your smartphone telemetry data, offering both 2D and 3D visualizations for better insights.

---

## Features

- **Visualization**: Displays telemetry data on 2D line plots and 3D animations.
- **Data Compatibility**: Works with telemetry data recorded via SensorLogger or similar apps (CSV format required).
- **Data Cleaning**: Ensures required columns are present and handles missing or invalid data.
- **Conversions**:
  - Speed converted from m/s to km/h.
  - Angular measurements converted from radians to degrees.
- **3D Animation**: Visualizes the movement of a block representing the device in 3D space.
- **Map Integration**: Displays the device's trajectory on a map using OpenStreetMap tiles.

---

## Prerequisites

### Required Libraries

Make sure you have the following Python libraries installed. You can install them using `pip` if needed:

```bash
pip install pandas numpy matplotlib scipy cartopy
```

### Library Guide

- **pandas**: For data manipulation and cleaning.
- **numpy**: For numerical operations.
- **matplotlib**: For plotting data and creating 3D visualizations.
- **mpl_toolkits**: Required for 3D plotting with Matplotlib.
- **scipy**: For handling quaternions and rotation calculations.
- **cartopy**: For geographic visualizations with maps.

---

## How to Use

1. **Prepare Your Data**:
   - Record telemetry data using **SensorLogger** or another app that supports CSV export.
   - Ensure the CSV file includes the required columns:
     - `seconds_elapsed`, `accelerometer_*`, `gyroscope_*`, `orientation_*`, `location_*`.

2. **Edit the Code**:
   - Open the `Telemetry Replay` script.
   - Replace the placeholder `csv_file` with the path to your CSV file:
     ```python
     csv_file = 'path/to/your/telemetry_data.csv'
     ```
   - Edit column names if your CSV format differs from the defaults.

3. **Run the Script**:
   - Execute the script to visualize the telemetry data. The visualizations include:
     - 3D animation of the device's position and orientation.
     - Line plots for speed, gyroscope, accelerometer, and orientation angles.
     - A map showing the device's trajectory.

4. **Enjoy Your Data Replay!**

---

### Example Output

- **3D View**: Shows the device's movement and orientation.
- **Line Plots**: Displays speed, orientation (yaw, pitch, roll), gyroscope, and accelerometer readings over time.
- **Map View**: Illustrates the GPS trajectory on an interactive map.

---

Feel free to report issues or suggest improvements. Enjoy exploring your telemetry data!
