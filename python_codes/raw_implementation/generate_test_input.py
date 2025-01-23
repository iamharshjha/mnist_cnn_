# import numpy as np
# from tensorflow import keras
# from keras.datasets import mnist
#
# # Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # Select the first image for this example
# image = x_train[0]  # First image in the training set
# print("Original image shape: ", image.shape)  # (28, 28)
#
# # Rescale the pixel values to the range [0, 255] (they are in [0, 1] initially)
# image = image.astype(np.uint8)  # Convert to unsigned 8-bit integers
#
#
# # Function to convert the image to hexadecimal format
# def convert_to_hex(image):
#     hex_values = []
#     for row in image:
#         hex_row = []
#         for pixel in row:
#             hex_value = f"{pixel:02X}"  # Convert each pixel to 2-digit hex
#             hex_row.append(hex_value)
#         hex_values.append(" ".join(hex_row))
#
#     return hex_values
#
#
# # Convert the image
# hex_image = convert_to_hex(image)
#
# # Output the hex values in the format for your Verilog testbench
# # We'll join the rows with newlines, so each row is on a new line in the text file.
# hex_image_str = "\n".join(hex_image)
#
# # Write the hex data to a text file (e.g., "mnist_image.hex")
# with open("mnist_image.hex", "w") as file:
#     file.write(hex_image_str)
#
# print("Hexadecimal values saved to 'mnist_image.hex'")
import numpy as np

# Load the .hex file (assumes each line represents a row of the matrix with space-separated hex values)
with open("mnist_image.hex", "r") as f:
    hex_values = []
    for line in f.readlines():
        # Split each line into individual hex values and convert them to integers
        hex_values.extend([int(x, 16) for x in line.strip().split()])

# Convert the list of hex values to a numpy array (assuming a 28x28 image, reshaping it into a 2D matrix)
image_data = np.array(hex_values).reshape((28, 28))

# Check the shape of the data (should be 28x28)
assert image_data.shape == (28, 28), f"Expected 28x28 matrix, but got shape {image_data.shape}"

# Flatten the 2D matrix into a 1D array for writing to a .mem file
flat_array = image_data.flatten()

# Write the flat array to a .mem file
with open("output_image.mem", "w") as f:
    for value in flat_array:
        # Write each hex value to the .mem file (in hex format)
        f.write(f"{value:02X}\n")  # Format as two-character hex values

print("Conversion complete!")
