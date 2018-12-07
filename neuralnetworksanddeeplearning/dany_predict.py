import pickle
import os.path
import numpy as np
import PIL.ImageOps
from PIL import Image
from neuralnetworksanddeeplearning.src import mnist_loader, network


def interpolate(value, origmin, origmax, targetmin, targetmax):
    # Figure out how 'wide' each range is
    leftSpan = origmax - origmin
    rightSpan = targetmax - targetmin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - origmin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return targetmin + (valueScaled * rightSpan)


def get_image():
    img = Image.open("digits.png")
    img.show()
    r, g, b, a = img.split()
    inverted_image = PIL.ImageOps.invert(r)
    # inverted_image.show()
    r2 = inverted_image
    pixels = r2.load()
    sumaAnterior = 0
    last_img = []
    for i in range(0, 623):
        sumaActual = 0
        for j in range(0, 128):
            sumaActual += pixels[i, j]
            # print(pixels[i,j])
        if sumaAnterior == 0 and sumaActual > 0:
            inicio = i
            print("Empieza en: " + str(i))
        elif (sumaAnterior > 0 and sumaActual == 0) or (i == 622 and sumaActual > 0):
            last_img = [inverted_image.crop((inicio, 0, i, 128))]
            # inverted_image.crop((inicio, 0, i, 128)).show()
            print("Termina en: " + str(i))
        sumaAnterior = sumaActual

    img_pixels = last_img[0].resize((28, 28))

    img_pixels = img_pixels.load()

    flattened = []
    for y in range(0, 28):
        for x in range(0, 28):
            flattened.append(img_pixels[x, y])

    flattened_np = [np.reshape(flattened, (784, 1)), 2]
    flattened_np[0] = flattened_np[0].astype(np.float32)
    print("interpolate 0-255 pixels to 0-1")
    for index, pixel in enumerate(flattened_np[0]):
        flattened_np[0][index] = interpolate(pixel, 0, 255, 0, 1)

    return flattened_np


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
if not os.path.isfile('pesos.pickle'):
    net = network.Network([784, 90, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
else:
    infile = open('pesos.pickle', 'rb')
    network = pickle.load(infile)
    print("Pickle loaded")
    image = get_image()
    print(network.feedforward(image[0]))
