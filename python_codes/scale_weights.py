import json
import numpy as np
import os

## to change the scaling factor go to line number 50 ##


def read_json_file(file_path):
    with open(file_path, "r") as fp:
        weights = json.load(fp)
    return weights


def find_max_abs_value(nested_list):
    """Recursively find the maximum absolute value in a multi-nested list."""
    max_val = float('-inf')
    for item in nested_list:
        if isinstance(item, list):
            max_val = max(max_val, find_max_abs_value(item))
        else:
            max_val = max(max_val, abs(item))
    return max_val


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


def process_all_json_files(input_dir, output_subdir):
    output_dir = os.path.join(input_dir, output_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            input_file_path = os.path.join(input_dir, file_name)
            weights = read_json_file(input_file_path)

            max_val = find_max_abs_value(weights)
            scale_factor = 127 / max_val  # Scaling to fit in the range -128 to 127
            scaled_weights = scale_nested_list(weights, scale_factor)

            print(f"Max value before scaling for {file_name}: {max_val}")

            output_file_path = os.path.join(output_dir, file_name)
            save_scaled_weights(scaled_weights, output_file_path)


def main():
    input_dir = 'weights'  # Directory containing input JSON files
    output_subdir = 'scaled_12bit_signed'  # Subdirectory to save scaled JSON files
    process_all_json_files(input_dir, output_subdir)


if __name__ == "__main__":
    main()
