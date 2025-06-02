import pandas as pd
import numpy as np
from PIL import Image

# File paths
image_path = r"C:/Users/Ritchie Strachan/Desktop/2024-25 Science Fair/Lake Winnipeg 03 Half size.png"
csv_path = r"C:/Users/Ritchie Strachan/Desktop/2024-25 Science Fair/Lake Winnipeg 03 Half size distances.csv"
output_csv = r"C:/Users/Ritchie Strachan/Desktop/2024-25 Science Fair/Processed Lake Winnipeg 03 Half size b.csv"

# Load the image
image = Image.open(image_path).convert("RGB")
pixels = np.array(image)

# Load the total values
df = pd.read_csv(csv_path)

# Check for 'Total' column
if "sum_distance" not in df.columns:
    raise ValueError("CSV does not contain a 'sum_distance' column.")

total_values = df["sum_distance"].values
value_index = 0

results = []

# Iterate through image pixels
for y in range(pixels.shape[0]):
    print(f"Processing row {y} of {pixels.shape[0]}", end='\r')
    for x in range(pixels.shape[1]):
        r, g, b = pixels[y, x]
        if (r, g, b) != (0, 0, 0):  # Non-black pixel
            if value_index >= len(total_values):
                raise ValueError("More non-black pixels than total values in the CSV.")
            results.append([x, y, total_values[value_index]])
            value_index += 1

# Confirm alignment
if value_index != len(total_values):
    raise ValueError(f"CSV contains more total values ({len(total_values)}) than non-black pixels ({value_index}).")

# Save results
output_df = pd.DataFrame(results, columns=["x", "y", "total_value"])
output_df.to_csv(output_csv, index=False)

print(f"\nProcessed image data saved to: {output_csv}")
