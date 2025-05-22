import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Bresenham's Line Algorithm with Distance Cap
def find_distance_bresenham(image_array, x0, y0, angle, max_dist=100):
    h, w = image_array.shape
    rad = np.deg2rad(angle)
    
    # Compute endpoint far along the given angle direction (limited to max_dist)
    x1 = int(x0 + max_dist * np.cos(rad))
    y1 = int(y0 + max_dist * np.sin(rad))
    
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    distance = 0

    while True:
        # Stop if max distance is exceeded
        if distance > max_dist:
            return max_dist

        # Out-of-bounds check
        if x < 0 or x >= w or y < 0 or y >= h:
            return max_dist

        # Check if we hit a black pixel
        if image_array[y, x] == 0:
            return distance

        # Bresenham's algorithm step
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

        distance += 1

# Load the image and convert to grayscale
Image.MAX_IMAGE_PIXELS = None  # Disable the pixel limit
image = Image.open("Desktop/Science Fair/Lake Winnipeg 03 Quarter size.png").convert("L")
image_array = np.array(image)

# Create DataFrame to store distances for debugging purposes
angles = range(0, 360, 10)  # Check every 10 degrees
df = pd.DataFrame(columns=[f"angle_{a}" for a in angles])

# Prepare the output image
distance_sum_image = np.zeros_like(image_array, dtype=np.float64)
height, width = image_array.shape

# Decide on a frequency to print progress updates
progress_frequency = 1  # Print progress every 100 rows

# Process each pixel
for y in range(height):
    # Print a progress message every 100 rows
    if y % progress_frequency == 0:
        print(f"Processing row {y+1} of {height}...")

    for x in range(width):
        if image_array[y, x] != 0:  # Skip black pixels (shorelines)
            distances = []
            for angle in angles:  # Cast rays in all directions
                distance = find_distance_bresenham(image_array, x, y, angle, max_dist=100)
                distances.append(distance)
            
            # Store the distances for debugging
            df.loc[len(df)] = distances
            
            # Sum distances for visualization
            distance_sum_image[y, x] = np.sum(distances)
        else:
            distance_sum_image[y, x] = 0

print("Processing complete. Normalizing and saving results...")

# Normalize the summed distances to 0–100 for visualization
normalized_image = distance_sum_image / (len(angles) * 100)

# Display the resulting heatmap
plt.imshow(normalized_image, cmap='inferno')
plt.colorbar(label="Normalized Sum of Distances (0–100)")
plt.title("Heatmap of Sum of Distances to Nearest Black Pixel")
plt.show()

plt.savefig("Desktop/Science Fair/Lake Winnipeg 03 Half size.png", dpi=300)  # Saves the figure

df.to_csv("Desktop/Science Fair/pixel_distance.csv",index=False)
print("Data saved to pixel_distances.csv")