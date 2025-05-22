import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from matplotlib.image import imread
import math
import random

##############################################################################
# GLOBAL TUNING VARIABLES
##############################################################################

NUM_DOTS = 5000         # how many red dots
MOVEMENT_SCALE = 0.01    # lower => slower, higher => faster
DOT_SIZE = 0.5           # dot size in the scatter plot

##############################################################################
# 1) LOAD WEATHER DATA & STATION INFO (Windy14 Logic)
##############################################################################

file_path = "/Users/ellastrachan/Desktop/Science Fair/Updated_Weather_Data_with_Coordinates.csv"
weather_data = pd.read_csv(file_path)

# Predefined station coordinates
station_coords = {
    "Great Falls": {"lat": 50.5222, "lon": -95.9772},
    "Victoria Beach": {"lat": 50.681274, "lon": -96.542453},
    "The Pas": {"lat": 53.9714, "lon": -101.0911},
    "Berens River": {"lat": 52.35, "lon": -97.03},
    "Gimli": {"lat": 50.602885, "lon": -96.983084},
    "Grand Rapids": {"lat": 53.189137, "lon": -99.256629},
    "Norway House": {"lat": 53.97, "lon": -97.85},
    "Fisher Branch": {"lat": 51.0834, "lon": -97.5545},
    "Pinawa": {"lat": 50.1772, "lon": -96.0647},
    "Cross Lake": {"lat": 54.5397, "lon": -98.0303},
    "Winnipeg": {"lat": 49.9167, "lon": -97.2494},
    "Island Lake": {"lat": 53.8572, "lon": -94.6536},
    "George Island": {"lat": 52.822417, "lon": -97.643281}
}

# Add latitude and longitude columns
weather_data["Latitude"] = weather_data["Station"].map(lambda x: station_coords[x]["lat"])
weather_data["Longitude"] = weather_data["Station"].map(lambda x: station_coords[x]["lon"])

# Normalize wind speed for color mapping
speed_max = weather_data["Speed"].max()
norm_speed = mcolors.Normalize(vmin=0, vmax=speed_max)
cmap_speed = plt.cm.viridis  # color map for wind speed

# Define a global grid for interpolation
grid_x, grid_y = np.linspace(-102, -94, 50), np.linspace(49.5, 55, 50)
grid_lon, grid_lat = np.meshgrid(grid_x, grid_y)

# Define the region to display
lat_min, lat_max = 50.311481783554, 53.96444429103665
lon_min, lon_max = -99.36544585392511, -96.1397058869483

##############################################################################
# 2) LOAD LAKE IMAGE
##############################################################################

lake_image_path = "/Users/ellastrachan/Desktop/Science Fair/Lake Winnipeg 03 tiny xxx.png"
# -- The UNMODIFIED array for logic:
original_lake_map = imread(lake_image_path)

# A copy for display (optional if you need it)
lake_image_for_plot = original_lake_map.copy()

# Image shape for bounding checks:
img_h, img_w = original_lake_map.shape[:2]

##############################################################################
# 3) WIND INTERPOLATION FUNCTIONS
##############################################################################

def compute_station_vectors(direction, length=0.05):
    """
    For station arrows, as in your original code.
    direction => we do (36 - dir)*10 - 90
    """
    angle_deg = (36 - direction) * 10 - 90
    dx = length * np.cos(np.radians(angle_deg))
    dy = length * np.sin(np.radians(angle_deg))
    return dx, dy

def compute_interpolated_vectors(direction, length=0.05):
    """
    For grid arrows, using already transformed direction:
    (36 - direction)*10 - 90 => angle in degrees
    """
    dx = length * np.cos(np.radians(direction))
    dy = length * np.sin(np.radians(direction))
    return dx, dy

def interpolate_wind_data(timestamp_data):
    """
    Interpolate Speed & Direction onto (grid_lon, grid_lat) using `griddata`.
    Then compute a fixed-length vector field (u_grid, v_grid).
    """
    points = np.array([timestamp_data["Longitude"], timestamp_data["Latitude"]]).T
    values_speed = timestamp_data["Speed"]
    # Transform direction
    values_direction = (36 - timestamp_data["Direction"]) * 10 - 90

    grid_speed = griddata(points, values_speed, (grid_lon, grid_lat), method="linear")
    grid_speed = np.nan_to_num(grid_speed)  # replace NaN w/ 0

    # Interpolate direction via trig
    grid_sin = griddata(points, np.sin(np.radians(values_direction)), (grid_lon, grid_lat), method="linear")
    grid_cos = griddata(points, np.cos(np.radians(values_direction)), (grid_lon, grid_lat), method="linear")
    grid_sin = np.nan_to_num(grid_sin)
    grid_cos = np.nan_to_num(grid_cos)
    grid_direction = np.degrees(np.arctan2(grid_sin, grid_cos))

    u_grid, v_grid = compute_interpolated_vectors(grid_direction)
    return u_grid, v_grid, grid_speed

##############################################################################
# 4) DOT MOVEMENT LOGIC: Move according to local wind, stop on black
##############################################################################

dot_positions = []   # each item: [lon, lat]
dot_stopped   = []   # bool
dots_scatter  = None # for plotting them

def find_nearest_index(lon_val, lat_val):
    """
    Finds the nearest index (r, c) in grid_lon/grid_lat to (lon_val, lat_val).
    """
    dist_sq = (grid_lon - lon_val)**2 + (grid_lat - lat_val)**2
    idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    return idx  # (row, col)

def get_displacement(speed_kph):
    """
    Convert wind speed (kph) to a displacement factor:
      2..4% drift + base scale => 15 KPH => ~3..6 px
    """
    drift = random.uniform(0.02, 0.04)
    base_pixels = 0.2  # so 15 KPH => ~3 px at 0.02
    return speed_kph * drift * base_pixels

def is_black_pixel(lon_val, lat_val):
    """
    Convert (lon_val, lat_val) -> pixel (x, y) in the ORIGINAL lake map.
    Return True if that pixel is black => [0,0,0].
    """
    if np.isnan(lon_val) or np.isnan(lat_val):
        return False

    denom_lon = (lon_max - lon_min)
    denom_lat = (lat_max - lat_min)
    if denom_lon == 0 or denom_lat == 0:
        return False

    x_pixel = (lon_val - lon_min) / denom_lon * (img_w - 1)
    y_pixel = (lat_max - lat_val) / denom_lat * (img_h - 1)

    if np.isnan(x_pixel) or np.isnan(y_pixel):
        return False

    x_int = int(round(x_pixel))
    y_int = int(round(y_pixel))

    if x_int < 0 or x_int >= img_w or y_int < 0 or y_int >= img_h:
        return False

    color = original_lake_map[y_int, x_int]
    if len(color) == 4:  # RGBA
        r, g, b, a = color
    else:
        r, g, b = color

    # If truly black => (0,0,0)
    return (r == 0 and g == 0 and b == 0)

def move_dots(u_grid, v_grid, grid_speed):
    """
    For each dot (lon, lat):
      - find nearest cell
      - read local wind
      - scale => displacement
      - update (lon, lat)
      - if black => stop
    """
    global dot_positions, dot_stopped

    for i in range(len(dot_positions)):
        if dot_stopped[i]:
            continue

        lon_val, lat_val = dot_positions[i]
        r_idx, c_idx = find_nearest_index(lon_val, lat_val)

        local_speed_kph = grid_speed[r_idx, c_idx]
        u_val = u_grid[r_idx, c_idx]
        v_val = v_grid[r_idx, c_idx]

        # If NaN or invalid => skip
        if np.isnan(local_speed_kph) or np.isnan(u_val) or np.isnan(v_val):
            continue

        # base factor from wind
        raw_scale = get_displacement(local_speed_kph) / 0.05
        # apply the global MOVEMENT_SCALE
        raw_scale *= MOVEMENT_SCALE

        dx = u_val * raw_scale
        dy = v_val * raw_scale
        if np.isnan(dx) or np.isnan(dy):
            continue

        new_lon = lon_val + dx
        new_lat = lat_val + dy
        if np.isnan(new_lon) or np.isnan(new_lat):
            continue

        # bounds check
        if new_lon < lon_min: new_lon = lon_min
        if new_lon > lon_max: new_lon = lon_max
        if new_lat < lat_min: new_lat = lat_min
        if new_lat > lat_max: new_lat = lat_max

        dot_positions[i] = [new_lon, new_lat]
        # stop if black
        if is_black_pixel(new_lon, new_lat):
            dot_stopped[i] = True

def init_dots(custom_starts=None):
    """
    Initialize dot positions. If `custom_starts` is provided, 
    it must be a list of [(lon1, lat1), (lon2, lat2), ...].
    Otherwise, pick random positions in [lon_min..lon_max, lat_min..lat_max],
    ensuring they do not start on black pixels.
    """
    global dot_positions, dot_stopped
    dot_positions = []
    dot_stopped = []

    if custom_starts is not None:
        for (start_lon, start_lat) in custom_starts:
            # Check if the custom start is valid
            if not is_black_pixel(start_lon, start_lat):
                dot_positions.append([start_lon, start_lat])
                dot_stopped.append(False)
    else:
        for _ in range(NUM_DOTS):
            valid = False
            while not valid:
                rnd_lon = random.uniform(lon_min, lon_max)
                rnd_lat = random.uniform(lat_min, lat_max)
                if not is_black_pixel(rnd_lon, rnd_lat):
                    dot_positions.append([rnd_lon, rnd_lat])
                    dot_stopped.append(False)
                    valid = True



##############################################################################
# 5) MATPLOTLIB FIGURE & ANIMATION (Slider-based)
##############################################################################

fig, ax = plt.subplots(figsize=(4, 6), dpi=100)
plt.subplots_adjust(bottom=0.2)

# Display lake_image_for_plot in the background
ax.imshow(
    lake_image_for_plot,  # purely for display, not used in logic
    extent=[lon_min, lon_max, lat_min, lat_max],
    aspect='auto'
)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_title("Wind Map + Red Dots")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

quiver_stations = None
quiver_interpolated = None
quadmesh = None
dots_scatter = None

timestamps = weather_data["Timestamp"].dropna().unique()
num_timestamps = len(timestamps)
current_frame = 0

def update(frame):
    global quiver_stations, quiver_interpolated, quadmesh, dots_scatter

    # Remove old objects
    if quiver_stations:
        quiver_stations.remove()
    if quiver_interpolated:
        quiver_interpolated.remove()
    if quadmesh:
        quadmesh.remove()
    if dots_scatter:
        dots_scatter.remove()

    # Extract data for current timestamp
    timestamp_data = weather_data[weather_data["Timestamp"] == timestamps[frame]]

    # 1) Plot station arrows
    u_stations, v_stations = compute_station_vectors(timestamp_data["Direction"])
    quiver_stations = ax.quiver(
        timestamp_data["Longitude"], timestamp_data["Latitude"],
        u_stations, v_stations,
        timestamp_data["Speed"], cmap=cmap_speed, norm=norm_speed,
        pivot="middle", zorder=6
    )

    # 2) Interpolate wind
    u_grid, v_grid, grid_speed = interpolate_wind_data(timestamp_data)

    # 3) Limit to region
    mask = (
        (grid_lat >= lat_min) & (grid_lat <= lat_max) &
        (grid_lon >= lon_min) & (grid_lon <= lon_max)
    )
    limited_u_grid = np.where(mask, u_grid, np.nan)
    limited_v_grid = np.where(mask, v_grid, np.nan)
    limited_grid_speed = np.where(mask, grid_speed, np.nan)

    # 4) Plot colored wind field
    quadmesh = ax.pcolormesh(
        grid_lon, grid_lat, limited_grid_speed,
        shading='auto', cmap=cmap_speed, norm=norm_speed, alpha=0.5, zorder=5
    )

    # 5) Plot quiver
    quiver_interpolated = ax.quiver(
        grid_lon[::2, ::2], grid_lat[::2, ::2],
        limited_u_grid[::2, ::2], limited_v_grid[::2, ::2],
        limited_grid_speed[::2, ::2],
        cmap=cmap_speed, norm=norm_speed,
        pivot="middle", alpha=0.7, zorder=10
    )

    # 6) Move dots
    move_dots(limited_u_grid, limited_v_grid, limited_grid_speed)

    # 7) Plot dots in red (use DOT_SIZE for bigger or smaller points)
    lon_vals = [p[0] for p in dot_positions]
    lat_vals = [p[1] for p in dot_positions]
    dots_scatter = ax.scatter(
        lon_vals, lat_vals,
        color="red",
        s=DOT_SIZE,      # <-- Adjust dot size
        marker="o",
        zorder=12
    )

    ax.set_title(f"Wind Map + Red Dots - {timestamps[frame]}")

# Create the slider
ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, "Time", 0, num_timestamps - 1, valinit=0, valstep=1)

def slider_update(val):
    global current_frame
    current_frame = int(slider.val)
    update(current_frame)
    fig.canvas.draw_idle()

slider.on_changed(slider_update)

# Keyboard
def on_key(event):
    global current_frame
    if event.key == "right" and current_frame < num_timestamps - 1:
        current_frame += 1
    elif event.key == "left" and current_frame > 0:
        current_frame -= 1
    slider.set_val(current_frame)
    update(current_frame)

fig.canvas.mpl_connect("key_press_event", on_key)

# Initialize the dots. You can pass a custom list of starts or leave None:
init_dots(custom_starts=None)

# Draw the first frame
update(0)

plt.show()




