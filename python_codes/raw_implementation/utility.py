import numpy as np
import math
import random

def RELU(image:list , shape : tuple):
    width = shape[0]
    height = shape[1]
    channels = shape[2]

    output = np.zeros((height , width , channels)) # creates an output matrix of all zeros\
    for y in range (height):
        for x in range(width):
            for z in range(channels):
                input = image[x][y][z]
                output[x][y][z] = input * (input>0)

    return output.tolist()

def SOFTMAX(image: list):
    # Find the maximum value in the image to improve numerical stability
    max_value = max(image)

    # Subtract the maximum value from each element in the image
    exps = [math.exp(pxi - max_value) for pxi in image]

    # Compute the sum of the exponentials
    expSum = sum(exps)

    # Compute the softmax output
    output = [ei / expSum for ei in exps]

    # print("output after SOFTMAX:")
    # print(output)
    return output

def MAXPOOLING(input_img: list, shape: tuple, poolSize: tuple):
    width, height, channels = shape
    pxsize, pysize = poolSize
    output = np.zeros((width // pxsize, height // pysize, channels))

    for y in range(0, height, pysize):
        for x in range(0, width, pxsize):
            for ch in range(channels):
                max_val = float('-inf')  # Initialize to negative infinity
                for py in range(pysize):
                    for px in range(pxsize):
                        if (x + px) < width and (y + py) < height:
                            max_val = max(max_val, input_img[y + py][x + px][ch])
                output[y // pysize][x // pxsize][ch] = max_val
    print("shape after the pooling layer:", np.array(output).shape)
    return output.tolist()

def flatten(inputImage: list) -> list:
    arr = np.array(inputImage)
    return arr.flatten().tolist()

def flatten_column_major(inputImage: list) -> list:
    channels = len(inputImage)  # Number of channels
    height = len(inputImage[0])  # Height of the image
    width = len(inputImage[0][0])  # Width of the image

    flattened = []

    # Iterate through each spatial position (y, x)
    for x in range(width):
        for y in range(height):
            for ch in range(channels):
                flattened.append(inputImage[ch][y][x])

    return flattened

def DENSE(inputimg: list, weights: list, shape: tuple) -> list:
    inlen, outlen = shape
    mulWeights = weights[0]
    biasWeights = weights[1]

    # print("weights:", mulWeights, " ")
    # print("Shape of weights:", np.array(mulWeights).shape)
    # print("bias:", biasWeights)
    # print("Shape of bias:", np.array(biasWeights).shape)
    # print(inlen , outlen)
    output = np.zeros(outlen)

    # Iterate over each output neuron
    for i in range(outlen):
        sum_value = 0  # To store the summation for the output neuron
        #print(f"Calculating output neuron {i}:")

        # Iterate over each input neuron
        for y in range(inlen):
            # Multiply the input neuron with the corresponding weight

            mul_value = inputimg[y] * mulWeights[y][i]
            sum_value += mul_value  # Add to the sum for the output neuron

            # Show the intermediate multiplication
            #print(f"  input[{y}] * weight[{y}][{i}] = {inputimg[y]} * {mulWeights[y][i]} = {mul_value}")

        # Add the bias for the output neuron
        sum_value += biasWeights[i]
        #print(f"  Adding bias: {biasWeights[i]}")

        # Set the final output value for the current neuron
        output[i] = sum_value
        #print(f"  Final output value for neuron {i}: {output[i]}")

    #print("Output after dense layer:", output)
    return output.tolist()


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

                    pointwise_output[x][y][chOut] += kr * px
    # pointwise_output += bias
    #print("output before adding the bias " , pointwise_output)
    pointwise_output += bias
    #print("output after adding the bias " , pointwise_output)
    print("shape after the convolution layer " , np.array(pointwise_output).shape)
    return pointwise_output.tolist()
