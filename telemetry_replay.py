# Telemetry Replay
# -----------------------------------------------------
# Developed by Gabi Rabelo (gabriela.rabelo@gmail.com)
# Replay your telemetry data using python
# 
# How to use:
# - Change [csv_file] to the path of your CSV file;
# - Edit column names if necessary;
# - Run the code;
# -----------------------------------------------------

# Import necessary Libs (you might need to inslatt some of these using pip)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

plt.style.use('ggplot')  # matplotlib visual style setting

# >>>>>>>>>>>>>>> EDIT HERE <<<<<<<<<<<<<<<<
csv_file = 'telemetry_data.csv'  # Path to your telemetry CSV file

plot_width = 16
plot_height = 9
# >>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<

# Read CSV
df = pd.read_csv(csv_file)

# >>>>>>>>>>>>>>>>> EDIT HERE only if necessary <<<<<<<<<<<<<<<<<<
required_columns = [
    'seconds_elapsed',
    'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
    'orientation_qx', 'orientation_qy', 'orientation_qz', 'orientation_qw',
    'location_speed',
    'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
    'orientation_yaw', 'orientation_pitch', 'orientation_roll',
    'location_latitude', 'location_longitude'  # Ensure location columns exist
]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Ensure necessary columns exist
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column in CSV: {col}")

# Clean latitude and longitude columns
df = df.dropna(subset=required_columns)
df = df[np.isfinite(df['location_latitude']) & np.isfinite(df['location_longitude'])]

# Convert speed to km/h
df['location_speed'] = df['location_speed'] * 3.6
df['orientation_yaw'] = np.degrees(df['orientation_yaw'])
df['orientation_pitch'] = np.degrees(df['orientation_pitch'])
df['orientation_roll'] = np.degrees(df['orientation_roll'])
df['gyroscope_x'] = np.degrees(df['gyroscope_x'])
df['gyroscope_y'] = np.degrees(df['gyroscope_y'])
df['gyroscope_z'] = np.degrees(df['gyroscope_z'])

# Function to calculate buffer limits for plotting
def calc_buffer(data, factor=0.1):
    """Calculate buffer limits for plot axes."""
    data_min, data_max = data.min(), data.max()
    buffer = (data_max - data_min) * factor
    return data_min - buffer, data_max + buffer

# Parse data
timestamps = df['seconds_elapsed'].values
accelerometer_data = df[['accelerometer_x', 'accelerometer_y', 'accelerometer_z']].values
quaternions = df[['orientation_qx', 'orientation_qy', 'orientation_qz', 'orientation_qw']].values
latitudes = df['location_latitude'].values
longitudes = df['location_longitude'].values

# Calculate velocity and position
time_diff = np.diff(timestamps, prepend=0)
velocity = np.cumsum(accelerometer_data * time_diff[:, None], axis=0)
position = np.cumsum(velocity * time_diff[:, None], axis=0)

# Create the figure and layout
fig = plt.figure(figsize=(plot_width, plot_height))

# Total sum of ratios should be proportional to the number of rows
height_ratios = [2, 2, 2, 2, 1, 2]  # Adjust these ratios to set the height of each row
gs = fig.add_gridspec(6, 9, height_ratios=height_ratios)  # Six equal rows

div_pos_h = 3
div_pos_w = 4

# 3D plot for the polygon (occupies the first two rows on the left)
div_pos_h = 3
ax3d = fig.add_subplot(gs[:div_pos_h, :div_pos_w], projection='3d')  # First 3 rows, 4 columns

ax3d.set_title("3D Position", fontsize=10)
ax3d.tick_params(labelsize=8)

# Add street map using OpenStreetMap tiles
tiler = cimgt.OSM()  # OpenStreetMap tiles
div_pos_h = 3
ax_map = fig.add_subplot(gs[div_pos_h:, :div_pos_w], projection=tiler.crs)

ax_map.set_title("Map View with Street Network", fontsize=10)

# Set map extent with fallback for empty or invalid data
if df.empty:
    ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # Global default extent
else:
    ax_map.set_extent([longitudes.min() - 0.001, longitudes.max() + 0.001,
                       latitudes.min() - 0.002, latitudes.max() + 0.002], crs=ccrs.PlateCarree())

ax_map.add_image(tiler, 15)  # Add street map tiles at higher zoom level (12)
scatter_map = ax_map.scatter([], [], c='purple', s=10, transform=ccrs.PlateCarree())


# Line plots (occupying the right side)
line_w = 1

div_pos_h = 3
ax_speed = fig.add_subplot(gs[0, div_pos_w:])  # First row, speed

ax_speed.set_title('Speed (km/h)', fontsize=10)
ax_speed.set_xlabel('Time (s)', fontsize=8)
ax_speed.set_ylabel('Speed (km/h)', fontsize=8)
ax_speed.set_xlim(timestamps[0], timestamps[-1])
ax_speed.set_ylim(*calc_buffer(df['location_speed']))
line_speed, = ax_speed.plot([], [], label='Speed', color='orangered')
ax_speed.legend(fontsize=8)
ax_speed.tick_params(axis='both', which='major', labelsize=8)

ax_orientation = fig.add_subplot(gs[1, div_pos_w:])  # Second row, orientation angles

ax_orientation.set_title('Yaw, Pitch, and Roll', fontsize=10)
ax_orientation.set_xlabel('Time (s)', fontsize=8)
ax_orientation.set_ylabel('Angle (degrees)', fontsize=8)
ax_orientation.set_xlim(timestamps[0], timestamps[-1])
if df.empty:
    ax_orientation.set_ylim(-180, 180)  # Default angle range
else:
    ax_orientation.set_ylim(
        *calc_buffer(df[['orientation_yaw', 'orientation_pitch', 'orientation_roll']].values.flatten())
    )
line_yaw, = ax_orientation.plot([], [], label='Yaw', linewidth=line_w, color='deepskyblue')
line_pitch, = ax_orientation.plot([], [], label='Pitch', linewidth=line_w, color='purple')
line_roll, = ax_orientation.plot([], [], label='Roll', linewidth=line_w, color='orangered')
ax_orientation.legend(fontsize=8)
ax_orientation.tick_params(axis='both', which='major', labelsize=8)

div_pos_h = 3
ax_gyroscope = fig.add_subplot(gs[2, div_pos_w:])  # Third row, gyroscope

ax_gyroscope.set_title('Gyroscope', fontsize=10)
ax_gyroscope.set_xlabel('Time (s)', fontsize=8)
ax_gyroscope.set_ylabel('Angular Velocity (degrees)', fontsize=8)
ax_gyroscope.set_xlim(timestamps[0], timestamps[-1])
ax_gyroscope.set_ylim(*calc_buffer(df[['gyroscope_x', 'gyroscope_y', 'gyroscope_z']].values.flatten()))
line_gyro_x, = ax_gyroscope.plot([], [], label='Gyroscope X', linewidth=line_w, color='deepskyblue')
line_gyro_y, = ax_gyroscope.plot([], [], label='Gyroscope Y', linewidth=line_w, color='purple')
line_gyro_z, = ax_gyroscope.plot([], [], label='Gyroscope Z', linewidth=line_w, color='orangered')
ax_gyroscope.legend(fontsize=8)
ax_gyroscope.tick_params(axis='both', which='major', labelsize=8)

div_pos_h = 3
ax_accelerometer = fig.add_subplot(gs[3, div_pos_w:])  # Fourth row, accelerometer

ax_accelerometer.set_title('Accelerometer', fontsize=10)
ax_accelerometer.set_xlabel('Time (s)', fontsize=8)
ax_accelerometer.set_ylabel('Acceleration (m/s²)', fontsize=8)
ax_accelerometer.set_xlim(timestamps[0], timestamps[-1])
ax_accelerometer.set_ylim(*calc_buffer(df[['accelerometer_x', 'accelerometer_y', 'accelerometer_z']].values.flatten()))
line_accel_x, = ax_accelerometer.plot([], [], label='Accelerometer X', linewidth=line_w, color='deepskyblue')
line_accel_y, = ax_accelerometer.plot([], [], label='Accelerometer Y', linewidth=line_w, color='purple')
line_accel_z, = ax_accelerometer.plot([], [], label='Accelerometer Z', linewidth=line_w, color='orangered')
ax_accelerometer.legend(fontsize=8)
ax_accelerometer.tick_params(axis='both', which='major', labelsize=8)

# Define polygon and faces
block_length, block_width, block_height = 0.2, 0.4, 0.04
block_corners = np.array([
    [0, 0, 0],
    [block_length, 0, 0],
    [block_length, block_width, 0],
    [0, block_width, 0],
    [0, 0, block_height],
    [block_length, 0, block_height],
    [block_length, block_width, block_height],
    [0, block_width, block_height],
]) - np.array([block_length / 2, block_width / 2, block_height / 2])
face_colors = ['lightcoral', 'springgreen', 'orangered', 'gold', 'deepskyblue', 'darkorchid']
block_faces = [[block_corners[j] for j in face] for face in [
    [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
    [3, 0, 4, 7], [4, 5, 6, 7], [0, 1, 2, 3]
]]
block_poly = Poly3DCollection(block_faces, facecolors=face_colors, edgecolor='k', alpha=0.8)
ax3d.add_collection3d(block_poly)
elapsed_time_text = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes, fontsize=8)

# Animation function
def update(frame):
    # Update 3D polygon and map scatter plot
    current_position = position[frame]
    current_quaternion = quaternions[frame]
    rotation = R.from_quat(current_quaternion)
    rotated_corners = rotation.apply(block_corners)
    updated_faces = [[corner + current_position for corner in face] for face in [
        [rotated_corners[j] for j in [0, 1, 5, 4]],
        [rotated_corners[j] for j in [1, 2, 6, 5]],
        [rotated_corners[j] for j in [2, 3, 7, 6]],
        [rotated_corners[j] for j in [3, 0, 4, 7]],
        [rotated_corners[j] for j in [4, 5, 6, 7]],
        [rotated_corners[j] for j in [0, 1, 2, 3]],
    ]]
    block_poly.set_verts(updated_faces)
    ax3d.set_xlim(current_position[0] - 0.5, current_position[0] + 0.5)
    ax3d.set_ylim(current_position[1] - 0.5, current_position[1] + 0.5)
    ax3d.set_zlim(current_position[2] - 0.5, current_position[2] + 0.5)
    elapsed_time_text.set_text(f"Elapsed Time: {timestamps[frame]:.0f}s")

    scatter_map.set_offsets(np.c_[longitudes[:frame], latitudes[:frame]])

    # Update line plots
    line_speed.set_data(timestamps[:frame], df['location_speed'][:frame])
    line_yaw.set_data(timestamps[:frame], df['orientation_yaw'][:frame])
    line_pitch.set_data(timestamps[:frame], df['orientation_pitch'][:frame])
    line_roll.set_data(timestamps[:frame], df['orientation_roll'][:frame])
    line_gyro_x.set_data(timestamps[:frame], df['gyroscope_x'][:frame])
    line_gyro_y.set_data(timestamps[:frame], df['gyroscope_y'][:frame])
    line_gyro_z.set_data(timestamps[:frame], df['gyroscope_z'][:frame])
    line_accel_x.set_data(timestamps[:frame], df['accelerometer_x'][:frame])
    line_accel_y.set_data(timestamps[:frame], df['accelerometer_y'][:frame])
    line_accel_z.set_data(timestamps[:frame], df['accelerometer_z'][:frame])

    return (block_poly, scatter_map, elapsed_time_text, line_speed, line_yaw, 
            line_pitch, line_roll, line_gyro_x, line_gyro_y, line_gyro_z, 
            line_accel_x, line_accel_y, line_accel_z)

# Animate
ani = FuncAnimation(fig, update, frames=len(position), interval=1000 * (timestamps[1] - timestamps[0]), blit=False)
plt.tight_layout()
plt.show()
