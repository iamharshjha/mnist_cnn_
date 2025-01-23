import numpy as np
import json
#from utility import *

def DEPTHWISE_SEP(inputImage: list, inputShape: tuple, layer:list , kernelShape: tuple, pointwise_shape: tuple) -> list:
    width = inputShape[0]
    height = inputShape[1]
    channelIn = inputShape[2]
    depthwise_kernel = layer[0]
    pointwise_kernel = layer[1]
    bias = layer[2]
    kxSize = kernelShape[0]
    kySize = kernelShape[1]
    channelOut = pointwise_shape[3]  # Output channels after pointwise convolution
    print("Depthwise Kernel Shape:", np.array(depthwise_kernel).shape)
    print("Pointwise Kernel Shape:", np.array(pointwise_kernel).shape)
    print(depthwise_kernel)
    print("##########")
    print(pointwise_kernel)
    print(bias)
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

                        depthwise_output[x][y][chIn] += kr * px
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
                    print("chin . chout and kr")

                    print(chIn , chOut , kr)
                    print("###")
                    print(px)
                    pointwise_output[x][y][chOut] += kr * px
    # pointwise_output += bias
    #print("output before adding the bias " , pointwise_output)
    pointwise_output += bias
    #print("output after adding the bias " , pointwise_output)
    print("shape after the convolution layer " , np.array(pointwise_output).shape)
    return pointwise_output.tolist()
def loadWeightsFromJson(path: str) -> list:
    with open(path, 'r') as fp:
        data = json.load(fp)
        fp.close()
    return data
def loadLayers():
    layer0_conv2d_weights = loadWeightsFromJson('../weights/scaled_8bit_signed/layer0_conv2d.json')
    layer2_conv2d_weights = loadWeightsFromJson('../weights/scaled_8bit_signed/layer2_conv2d.json')
    layer5_dense_weights = loadWeightsFromJson('../weights/scaled_8bit_signed/layer5_dense.json')
    return layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights

    return layer0_conv2d_weights, layer2_conv2d_weights, layer5_dense_weights
def layer1():


    # Define the input values as provided
    data_out = np.array([
        1, 1, 1, 0, 0,  # First row
        0, 0, 0, 0, 0,  # Second row
        0, 0, 0, 0, 0,  # Third row
        0, 0, 0, 0, 0,  # Fourth row
        0, 0, 0, 0, 0  # Fifth row
    ])

    # Reshape the 1D array into a 5x5 matrix
    data_matrix = data_out.reshape((5, 5))
    data_matrix_expanded = np.expand_dims(data_matrix, -1)
    testModel = data_matrix_expanded.reshape([1, 5, 5, -1])
    # Print the 5x5 matrix
    print("5x5 Matrix:")
    print(data_matrix)
    layer0_weights, layer2_weights, layer5_weights = loadLayers()
    ans = DEPTHWISE_SEP(testModel[0] , (5 ,5,1), layer2_weights ,  (5,5,3,3) , (1,1,3,9))


def main():
    layer1()
    # layer0_weights, layer2_weights, layer5_weights = loadLayers()
    # dep = layer2_weights[0]
    # pw = layer2_weights[1]
    # bias =layer2_weights[2]
    # print(dep)
    # print("########################################")
    # print(pw)
    # print("########################################")
    # print(bias)
    # print("******************************************")
    # print(pw[0][0][1][0])

if __name__ == "__main__":
    main()