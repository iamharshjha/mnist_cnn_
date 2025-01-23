from numpy.ma.core import shape
from tensorflow import keras
import numpy as np
import json

#from python_codes.extract_weight import layer5_dense_weights


def _kerasToList(weight: list) -> list:
    dep = weight[0]
    point = weight[1]
    bias = weight[2]
    return [dep.numpy().tolist(), point.numpy().tolist(),bias.numpy().tolist()]
def _kerasToListdense(weight: list) -> list:
    kernel = weight[0]
    bias = weight[1]
    return [kernel.numpy().tolist(),bias.numpy().tolist()]


def _saveListAsJson(listToSave: list, name: str):
    with open(name, "w") as fp:
        json.dump(listToSave, fp)
        fp.close()

def main():
    model = keras.models.load_model('fine_tuned_model.h5', compile=True)
    model.summary()

    layer0_conv2d = model.layers[0]
    layer2_conv2d = model.layers[2]
    layer5_dense = model.layers[5]



    layer0_conv2d_weights = layer0_conv2d.weights
    layer2_conv2d_weights = layer2_conv2d.weights
    layer5_dense_weights = layer5_dense.weights


    layer0_conv2d_weights_list = _kerasToList(layer0_conv2d_weights)
    _saveListAsJson(layer0_conv2d_weights_list, './weights/scaled_8bit_signed1/layer0_conv2d.json')
    layer2_conv2d_weights_list = _kerasToList(layer2_conv2d_weights)
    _saveListAsJson(layer2_conv2d_weights_list, './weights/scaled_8bit_signed1/layer2_conv2d.json')

    layer5_dense_weights_list = _kerasToListdense(layer5_dense_weights)
    _saveListAsJson(layer5_dense_weights_list, "./weights/scaled_8bit_signed1/layer5_dense.json")

if __name__ == "__main__":
    main()
