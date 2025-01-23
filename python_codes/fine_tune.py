import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist


# Step 1: Extract Weights
def extract_weights(model_path):
    model = load_model(model_path)  # Load your pre-trained model
    layer_weights = {}

    for layer in model.layers:
        # Get weights for each layer
        weights = layer.get_weights()
        if weights:  # Only consider layers that have weights (like Conv, Dense layers)
            layer_weights[layer.name] = weights

    return layer_weights


# Step 2: Quantize Weights
def quantize_weights(layer_weights):
    quantized_weights = {}

    for layer_name, weights_list in layer_weights.items():
        quantized_layer_weights = []

        for weight_matrix in weights_list:
            # Find the max absolute value in the weight matrix
            max_val = np.max(np.abs(weight_matrix))

            # Scale the weights to fit within the signed 8-bit integer range (-128 to 127)
            scale_factor = 127 / max_val
            quantized_matrix = np.round(weight_matrix * scale_factor).astype(np.int8)

            quantized_layer_weights.append(quantized_matrix)

        quantized_weights[layer_name] = quantized_layer_weights

    return quantized_weights


# Step 3: Fine-Tune the Model
def fine_tune_model(model_path, quantized_weights, x_train, y_train, epochs=5, batch_size=32):
    # Load the model
    model = load_model(model_path)

    # Set quantized weights back into the model
    for layer_name, quantized_layer_weights in quantized_weights.items():
        layer = model.get_layer(layer_name)
        layer.set_weights(quantized_layer_weights)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fine-tune the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    return model, history


# Step 4: Save the Fine-Tuned Model
def save_model(model, model_path):
    model.save(model_path)


# Step 5: Main Function to Execute the Full Workflow
def main():
    model_path = 'trained_model.h5'  # Path to your pre-trained model

    # Extract the weights from the original model
    weights = extract_weights(model_path)

    # Quantize the weights to signed 8-bit integers
    quantized_weights = quantize_weights(weights)

    # Load MNIST dataset
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0

    # Fine-tune the model with the quantized weights
    fine_tuned_model, history = fine_tune_model(model_path, quantized_weights, x_train, y_train)

    # Save the fine-tuned model
    save_model(fine_tuned_model, 'fine_tuned_model.h5')

    # Optionally, evaluate the fine-tuned model on test data
    # (x_test, y_test) = mnist.load_data()[1]  # Use this if you want to evaluate on the test set
    # x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
    # test_loss, test_acc = fine_tuned_model.evaluate(x_test, y_test)
    # print(f"Test accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
