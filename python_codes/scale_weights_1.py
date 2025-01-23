import json
import numpy as np


def read_json_file(file_path):
    with open(file_path, "r") as fp:
        weights = json.load(fp)
    return weights


def find_max_abs_value_per_channel(nested_list):
    """Find the maximum absolute value per channel in a multi-nested list."""
    if isinstance(nested_list[0][0], list):
        # Handle case for 2D matrices (e.g., weights)
        max_vals = []
        for channel in nested_list:
            max_val = find_max_abs_value(channel)
            max_vals.append(max_val)
        return max_vals
    else:
        # Handle case for 1D arrays (e.g., biases)
        return [find_max_abs_value(nested_list)]


def find_max_abs_value(nested_list):
    """Recursively find the maximum absolute value in a list."""
    max_val = float('-inf')
    for item in nested_list:
        if isinstance(item, list):
            max_val = max(max_val, find_max_abs_value(item))
        else:
            max_val = max(max_val, abs(item))
    return max_val


def scale_nested_list_per_channel(nested_list, scale_factors):
    """Recursively scale the values in a multi-nested list per channel."""
    scaled_list = []
    if isinstance(nested_list[0][0], list):
        # Handle case for 2D matrices (e.g., weights)
        for channel, scale_factor in zip(nested_list, scale_factors):
            scaled_channel = scale_nested_list(channel, scale_factor)
            scaled_list.append(scaled_channel)
    else:
        # Handle case for 1D arrays (e.g., biases)
        scale_factor = scale_factors[0]
        scaled_list = scale_nested_list(nested_list, scale_factor)
    return scaled_list


def scale_nested_list(nested_list, scale_factor):
    """Recursively scale the values in a multi-nested list."""
    scaled_list = []
    for item in nested_list:
        if isinstance(item, list):
            scaled_list.append(scale_nested_list(item, scale_factor))
        else:
            scaled_list.append(int(np.round(item * scale_factor)))
    return scaled_list


def save_scaled_weights(scaled_weights, file_path):
    with open(file_path, "w") as fp:
        json.dump(scaled_weights, fp, indent=2)


def main():
    file_paths = ['weights/layer0_conv2d.json', 'weights/layer2_conv2d.json', 'weights/layer5_dense.json']  # Add all your layer paths here

    for file_path in file_paths:
        weights = read_json_file(file_path)

        max_vals_per_channel = find_max_abs_value_per_channel(weights)
        scale_factors = [127 / max_val for max_val in max_vals_per_channel]
        scaled_weights = scale_nested_list_per_channel(weights, scale_factors)

        print(f"Max values per channel before scaling: {max_vals_per_channel}")

        scaled_file_path = file_path.replace('weights', 'weights/scaled_8bit_signed')
        save_scaled_weights(scaled_weights, scaled_file_path)


if __name__ == "__main__":
    main()
