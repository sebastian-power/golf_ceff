import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = "image.png"  # Update this path if needed
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image to get the dimensions
plt.imshow(image_rgb)
plt.show()

# Define the known axis limits from the graph
x_min, x_max = 0, 1.7e5  # Limits for Reynolds number (Re)
y_min, y_max = 0, 0.6  # Limits for Drag coefficient (CD)

# Define colour range to isolate the blue points (e.g., for golf ball data points)
lower_blue = np.array([0, 0, 200])  # Lower bound for blue
upper_blue = np.array([100, 100, 255])  # Upper bound for blue

# Create a mask for blue points
mask = cv2.inRange(image_rgb, lower_blue, upper_blue)

# Find contours of the blue points
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract (x, y) coordinates of each contour
points = []
for contour in contours:
    # Get the centroid of each contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        points.append((cx, cy))

# Convert (x, y) pixel coordinates to data values
data_points = []
img_height, img_width = image_rgb.shape[:2]

for px, py in points:
    # Map x (Re) from pixel to data range
    x_val = x_min + (px / img_width) * (x_max - x_min)
    # Map y (CD) from pixel to data range (invert y-axis as images start from the top)
    y_val = y_max - (py / img_height) * (y_max - y_min)
    data_points.append((x_val, y_val))

# Plot extracted points for verification
extracted_x, extracted_y = zip(*data_points)
plt.scatter(extracted_x, extracted_y, color="blue", label="Extracted Data Points")
plt.xlabel("Re")
plt.ylabel("CD")
plt.legend()
plt.show()

# Print extracted data
for re, cd in sorted(data_points):
    print(f"Re: {re:.2e}, CD: {cd:.2f}")
