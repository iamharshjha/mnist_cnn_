import numpy as np
import json
from utility import *
from tensorflow import keras

def loadLayers():
    layer0_conv2d_weights = loadWeightsFromJson('../weights/layer0_conv2d.json')
    layer2_conv2d_weights = loadWeightsFromJson('../weights/layer2_conv2d.json')
    layer5_dense_weights = loadWeightsFromJson('../weights/layer5_dense.json')
    return layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights

    return layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights
def loadWeightsFromJson(path: str) -> list:
    with open(path, 'r') as fp:
        data = json.load(fp)
        fp.close()
    return data


def loadModel():
    model = keras.models.load_model('../trained_model.h5', compile=True)
    return model

def testData():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_train = x_train.astype("float32") / 255 - 0.5
    # x_test = x_test.astype("float32") / 255 - 0.5
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    testModel = x_test.reshape([1, 28, 28, -1])
    return testModel ,y_test


def expectedAnswer(model, testModel):
    expected = model.predict(testModel)
    print(f"expected: {expected}")


def actualAnswer(layer0_conv2d_weights, layer2_conv2d_weights,layer5_dense_weights , testModel):

    layer0Out = DEPTHWISE_SEP(testModel[0], (28, 28, 1), layer0_conv2d_weights ,  (5,5,1,1) , (1,1,1,3))

    layer0ReLU = RELU(layer0Out, (24, 24, 3))

    layer1Out = MAXPOOLING(layer0ReLU, (24, 24, 3), (2, 2))

    layer2Out = DEPTHWISE_SEP(layer1Out, (12, 12, 3), layer2_conv2d_weights , (5,5,3,3) , (1,1,3,9))

    layer2ReLU = RELU(layer2Out, (8, 8, 9))

    layer3Out = MAXPOOLING(layer2ReLU, (8, 8, 9), (2, 2))

    layer4Flatten = flatten(layer3Out)
    layer5dense = DENSE(layer4Flatten , layer5_dense_weights , (144,10))

    output = SOFTMAX(layer5dense)
    max_index = output.index(max(output))
    #print("the model predicted output is: " , max_index)

    return max_index




def main():

    model = loadModel()
    n = 10000
    cnt = 0
    #layer0_weights , layer2_weights , layer5_weights , layer6_weights = loadLayers()
    layer0_weights, layer2_weights, layer5_weights = loadLayers()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train/1
    x_test = x_test/1
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    for i in range(n):
        #m = 5
        testModel = x_test[i].reshape([1, 28, 28, -1])
        #testmodel, ans = testData()

        ans = actualAnswer(layer0_weights, layer2_weights, layer5_weights , testModel)
        #print("actual answer:" ,y_test[i])
        if (y_test[i] == ans):
            cnt+=1
        print(f"Processing i = {i}")
    print("accuracy ->" , (cnt/n)*100)



if __name__ == "__main__":
    main()
