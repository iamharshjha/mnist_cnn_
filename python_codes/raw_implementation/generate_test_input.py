import numpy as np
from tensorflow.keras.datasets import mnist


def mnist_to_mem(image_array, output_path):
    # Ensure the image is 28x28 grayscale
    img_array = image_array.astype(np.uint8)

    # Flatten the array
    flat_array = img_array.flatten()

    # Convert to hex format
    hex_values = [f"{val:02X}" for val in flat_array]

    # Write to .mem file
    with open(output_path, "w") as f:
        for hex_val in hex_values:
            f.write(hex_val + "\n")

    print(f"Conversion complete! Data saved to {output_path}")


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select the first image for this example
image = x_train[2]

# Convert and save
mnist_to_mem(image, "1.mem")
