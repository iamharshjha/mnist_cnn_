import os
import json


def read_json_file(file_path):
    """Read and load the JSON content from a file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def flatten_and_convert_to_hex(nested_list):
    """Flatten the nested list and convert integers to their 8-bit signed hex representation."""
    flat_list = []

    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                flatten(item)
            else:
                # Convert to signed 8-bit integer and then to 2's complement hex
                hex_value = format(item & 0xFF, '02X')
                flat_list.append(hex_value)

    flatten(nested_list)
    return flat_list


def write_to_mem_file(hex_values, file_path):
    """Write the flattened and hex-converted values to a .mem file."""
    with open(file_path, "w") as f:
        for hex_value in hex_values:
            f.write(hex_value + '\n')


def process_json_files(input_dir, output_dir):
    """Process all JSON files in the given directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(input_dir, file_name)
            mem_file_path = os.path.join(output_dir, file_name.replace('.json', '.mem'))

            # Step 1: Read the JSON file
            data = read_json_file(json_file_path)

            # Step 2: Flatten the data and convert to hex
            hex_values = flatten_and_convert_to_hex(data)

            # Step 3: Write to .mem file
            write_to_mem_file(hex_values, mem_file_path)

            print(f"Processed {file_name} and saved as {mem_file_path}")


def main():
    input_dir = '../weights/scaled_8bit_signed'  # Directory containing JSON files
    output_dir = '../weights/mem/'  # Directory to save .mem files

    process_json_files(input_dir, output_dir)


if __name__ == "__main__":
    main()
