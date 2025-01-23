def main():
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers
    #Loading the Mnist dataset

    (x_train , y_train) ,(x_test , y_test) = tf.keras.datasets.mnist.load_data()

    #check the shapes of the dataset
    print("training shape" , x_train.shape)

    print("testing shape" , y_test.shape)

    # checking the datatypes

    print(x_train.dtype)

    x_train = x_train/255
    x_test  = x_test/255
    def convert_float32(d):
        return d.astype("float32")
    x_train = convert_float32(x_train)
    x_test = convert_float32(x_test)
    print("after normalizing the data" ,x_train.dtype)

    x_train = np.expand_dims(x_train , -1)
    x_test = np.expand_dims(x_test , -1)
    print("added another dimension " , x_train.shape)

    #converting to hot encoded labels

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    input_shape = (28,28,1)
    filter = 3
    stride = 1
    ks = (5,5)
    model = keras.Sequential(
        [
            tf.keras.layers.SeparableConv2D(
                filters=filter,  # Number of filters
                kernel_size=ks,  # Kernel size
                strides=stride,  # Stride size
                activation='relu',  # Activation function
                #padding='valid',  # Padding
                input_shape=(28, 28, 1)  # Input shape for MNIST
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Pooling layer
            tf.keras.layers.SeparableConv2D(
                filters=filter * 3,
                kernel_size=ks,
                strides=stride,
                activation='relu',
                #padding='valid'
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),  # Flatten the feature maps
            tf.keras.layers.Dense(
                units=10,  # Number of units in Dense layer
                activation='softmax'
            )
            # tf.keras.layers.Dropout(0.5),  # Dropout for regularization
            # tf.keras.layers.Dense(
            #     units=10,  # Number of output units (classes)
            #     activation='softmax'
            # )
        ]

    )
    model.summary()

    batch_size = 16
    epochs = 10

    model.compile(loss = "categorical_crossentropy" , optimizer = "adam" ,metrics = ["accuracy"])
    model.fit(x_train , y_train , batch_size = batch_size , epochs = epochs )

    score = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save('trained_model.h5', save_format='h5')
    model.save_weights("trained_model.weights.h5")

if __name__ == "__main__":
    main()
