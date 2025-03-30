import numpy as np
import matplotlib.pyplot as plt

# Define the size of the image
width, height = 1600, 1600
image = np.zeros((height, width, 3), dtype=np.uint8)

# Golden Ratio (phi)
phi = (1 + np.sqrt(5)) / 2  # Golden Ratio

# Equation parameters
G = 6.67430e-11  # Gravitational constant
c = 3.00e8      # Speed of light

def universe_equation(x, y):
	# Example function inspired by the equation components
	density = np.sin(x * y) * np.cos(y / (x + 1))
	feedback = np.sin(x) * np.cos(y)
	ecosystem = np.exp(-((x - width / 2) ** 2 + (y - height / 2) ** 2) / 10000)
	biodiversity = np.sin(np.sqrt(x**2 + y**2))
	consciousness = np.tanh((x - width / 2) * (y - height / 2) / 10000)

	# Incorporate the Golden Ratio into the function
	golden_influence = np.sin(x * phi) * np.cos(y * phi)  # Golden Ratio influence

	# Combine components into a single value
	value = density + feedback + ecosystem + biodiversity + consciousness + golden_influence
	return value

def normalize(value, min_val, max_val):
	if min_val == max_val:
		return 0.5  # Avoid division by zero
	return (value - min_val) / (max_val - min_val)

# We need to loop through all pixels, but first, we calculate the min and max values
# to normalize all pixels in the image.
min_val = float('inf')
max_val = float('-inf')

# First pass to find min and max
for x in range(width):
	for y in range(height):
		value = universe_equation(x, y)
		min_val = min(min_val, value)
		max_val = max(max_val, value)

# Second pass to normalize and generate the image
for x in range(width):
	for y in range(height):
		value = universe_equation(x, y)
		normalized_val = normalize(value, min_val, max_val)

		# Map the normalized value to RGB channels
		red = int(255 * np.sin(x / 100 + normalized_val + phi))  # Red channel influenced by x, normalized value, and golden ratio
		green = int(255 * np.cos(y / 100 + normalized_val + phi))  # Green channel influenced by y, normalized value, and golden ratio
		blue = int(255 * np.tanh((x - width / 2) * (y - height / 2) / 10000 + phi))  # Blue channel influenced by distance from center and golden ratio

		# Clamp the RGB values to ensure they are within the valid range [0, 255]
		red = np.clip(red, 0, 255)
		green = np.clip(green, 0, 255)
		blue = np.clip(blue, 0, 255)

		# Assign color values to the image
		image[y, x] = [red, green, blue]

# Save the image as a PNG file
plt.imsave('univital_theory_with_golden_ratio.png', image)

print("Image saved as 'univital_theory_with_golden_ratio.png'")
