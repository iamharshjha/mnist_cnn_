from tensorflow import keras

model = keras.models.load_model("./trained_model.h5" , compile = True)
for i in range(0,9):
    print(model.layers[i])
    print(i)