import numpy as np
import json
from utility import *
from tensorflow import keras

def loadLayers():
    layer0_conv2d_weights = loadWeightsFromJson('../weights/scaled_8bit_signed/layer0_conv2d.json')
    layer2_conv2d_weights = loadWeightsFromJson('../weights/scaled_8bit_signed/layer2_conv2d.json')
    layer5_dense_weights = loadWeightsFromJson('../weights/scaled_8bit_signed/layer5_dense.json')
    return layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights

# def loadLayers():
#     layer0_conv2d_weights = loadWeightsFromJson('../weights/layer0_conv2d.json')
#     layer2_conv2d_weights = loadWeightsFromJson('../weights/layer2_conv2d.json')
#     layer5_dense_weights = loadWeightsFromJson('../weights/layer5_dense.json')
#     return layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights

    #return layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights
def loadWeightsFromJson(path: str) -> list:
    with open(path, 'r') as fp:
        data = json.load(fp)
        fp.close()
    return data
def DEPTHWISE_SEPR(inputImage: list, inputShape: tuple, layer:list , kernelShape: tuple, pointwise_shape: tuple) -> list:
    width = inputShape[0]
    height = inputShape[1]
    channelIn = inputShape[2]
    depthwise_kernel = layer[0]
    pointwise_kernel = layer[1]
    bias = layer[2]
    kxSize = kernelShape[0]
    kySize = kernelShape[1]
    channelOut = pointwise_shape[3]  # Output channels after pointwise convolution
    # print("Depthwise Kernel Shape:", np.array(depthwise_kernel).shape)
    # print("Pointwise Kernel Shape:", np.array(pointwise_kernel).shape)
    # print(depthwise_kernel)
    # print("##########")
    # print(pointwise_kernel)
    # print(bias)
    # Step 1: Depthwise Convolution (each input channel has its own kernel)
    depthwise_output = np.zeros((width - kxSize + 1, height - kySize + 1, channelIn))  # output of depthwise conv

    for y in range(height - kySize + 1):
        for x in range(width - kxSize + 1):
            for chIn in range(channelIn):
                for ky in range(kySize):
                    for kx in range(kxSize):
                        if ((y + ky) >= height) or ((x + kx) >= width):
                            continue
                        # Kernel value at [kx, ky, chIn, chOut]
                        kr = depthwise_kernel[kx][ky][chIn][0]

                        # Input pixel at [x + kx, y + ky, chIn]
                        #print("kernel " , kr)
                        px = inputImage[x + kx][y + ky][chIn]
                        #print("input image region " , px)
                        # Add to the depthwise output for chIn
                        # print(f"kr: {kr}, type: {type(kr)}")
                        # print(f"px: {px}, type: {type(px)}")

                        depthwise_output[x][y][chIn] += int(round((kr * px) / 32))
                        #print("output:" , depthwise_output[x][y][chIn])
    # Step 2: Pointwise Convolution (1x1 convolution on depthwise output)
    pointwise_output = np.zeros((width - kxSize + 1, height - kySize + 1, channelOut))  # output of pointwise conv
    # print("output after depthwise conv:")
    # # print(depthwise_output)
    # print("Pointwise Kernel Shape:", np.array(pointwise_kernel).shape)

    for y in range(depthwise_output.shape[0]):
        for x in range(depthwise_output.shape[1]):
            for chOut in range(channelOut):
                for chIn in range(channelIn):
                    # Apply 1x1 pointwise convolution (mix all input channels)

                    kr = pointwise_kernel[0][0][chIn][chOut]
                    px = depthwise_output[x][y][chIn]

                    pointwise_output[x][y][chOut] += int(round((kr * px) / 32))
    # pointwise_output += bias
    #print("output before adding the bias " , pointwise_output)
    pointwise_output += bias
    #print("output after adding the bias " , pointwise_output)
    # print("shape after the convolution layer " , np.array(pointwise_output).shape)
    return pointwise_output.tolist()

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

def truncate_to_20bit(x):
    """
    Recursively traverse nested lists and truncate each number
    to the range of a 20-bit signed integer (-524288 to 524287).
    """
    if isinstance(x, list):
        return [truncate_to_20bit(item) for item in x]
    else:
        # Convert the number to an integer (if needed)
        x_int = int(x)
        # Clamp the value to the range [-524288, 524287] (20-bit signed integer)
        return max(-524288, min(524287, x_int))


def actualAnswer(layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights, testModel):

    layer0Out = DEPTHWISE_SEPR(testModel[0], (28, 28, 1), layer0_conv2d_weights, (5, 5, 1, 1), (1, 1, 1, 3))
    #layer0Out = truncate_to_20bit(layer0Out)

    print("output of first convolution (truncated to 20-bit): \n", layer0Out)

    layer0ReLU = RELU(layer0Out, (24, 24, 3))
    layer1Out = MAXPOOLING(layer0ReLU, (24, 24, 3), (2, 2))
    print("maxpooling")
    print(layer1Out)
    layer2Out = DEPTHWISE_SEPR(layer1Out, (12, 12, 3), layer2_conv2d_weights, (5, 5, 3, 3), (1, 1, 3, 9))
    layer2Out = np.array(layer2Out)
    layer2Out = np.round(layer2Out / 32).astype(int).tolist()
    #layer2Out = truncate_to_20bit(layer2Out)
    print("output after second convolution (truncated to 20-bit): \n", layer2Out)

    layer2ReLU = RELU(layer2Out, (8, 8, 9))
    layer3Out = MAXPOOLING(layer2ReLU, (8, 8, 9), (2, 2))

    layer4Flatten = flatten(layer3Out)

    layer5dense = DENSE(layer4Flatten, layer5_dense_weights, (144, 10))
    layer5dense = np.array(layer5dense)
    layer5dense = np.round(layer5dense / 32).astype(int).tolist()
    #layer5dense = truncate_to_20bit(layer5dense)
    print("output of dense layer (truncated to 20-bit): \n", layer5dense)

    output = SOFTMAX(layer5dense)
    max_index = output.index(max(output))

    return max_index




def main():

    model = loadModel()
    n = 1
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
        testModel = x_train[101].reshape([1, 28, 28, -1])
        #testmodel, ans = testData()

        ans = actualAnswer(layer0_weights, layer2_weights, layer5_weights , testModel)
        print("actual answer:" ,y_train[101])
        print("predicted answer , " ,ans)
        if (y_train[101] == ans):
            cnt+=1
        #print(f"Processing i = {i}")
    print("accuracy ->" , (cnt/n)*100)



if __name__ == "__main__":
    main()
