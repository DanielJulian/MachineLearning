import pickle
import os.path
import numpy as np
import PIL.ImageOps
from PIL import ImageDraw
from PIL import Image
from neuralnetworksanddeeplearning.src import mnist_loader, network


def get_predicted_number(prediction_array):
    array = prediction_array.tolist()
    return str(array.index(max(array)))


def interpolate(value, origmin, origmax, targetmin, targetmax):
    # Figure out how 'wide' each range is
    leftSpan = origmax - origmin
    rightSpan = targetmax - targetmin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - origmin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return targetmin + (valueScaled * rightSpan)


# Retorna lista de listas, donde cada lista tiene 2 tuplas, la primera indicando el inicio y las segunda el final del
# digito.
def get_digits_coords(inverted_image):
    digits_coords = []
    pixels = inverted_image.load()
    sumaAnterior = 0
    for i in range(0, 623):
        sumaActual = 0
        for j in range(0, 128):
            sumaActual += pixels[i, j]
        if sumaAnterior == 0 and sumaActual > 0:
            inicio = i
        elif (sumaAnterior > 0 and sumaActual == 0) or (i == 622 and sumaActual > 0):
            digits_coords.append([(inicio, 0), (i, 127)])  # 127 hardcodeado, es la altura de la imagen
        sumaAnterior = sumaActual
    return digits_coords


def get_np_digit(inverted_image, coords):
    digit_image = inverted_image.crop((coords[0][0], coords[0][1], coords[1][0], coords[1][1]))
    img_pixels = digit_image.resize((28, 28))
    img_pixels = img_pixels.load()

    flattened = []
    for y in range(0, 28):
        for x in range(0, 28):
            flattened.append(img_pixels[x, y])

    flattened_np = np.reshape(flattened, (784, 1))
    flattened_np = flattened_np.astype(np.float32)
    # Interpolate 0-255 pixels to 0-1 scale
    for index, pixel in enumerate(flattened_np):
        flattened_np[index] = interpolate(pixel, 0, 255, 0, 1)
    return flattened_np


img = Image.open("digits.png")
r, g, b, a = img.split()
inverted_image = PIL.ImageOps.invert(r)
digit_coords = get_digits_coords(inverted_image)

if (not os.path.isfile('pesos.pickle')):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 90, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
else:
    infile = open('pesos.pickle', 'rb')
    net = pickle.load(infile)
    print("Pickle loaded")

for coords in digit_coords:
    digit = get_np_digit(inverted_image, coords)
    predicted_number = get_predicted_number(net.feedforward(digit))
    print("Predicted number : ", predicted_number)
    draw = ImageDraw.Draw(img)
    draw.text((coords[1][0]-50, 0), predicted_number, (0, 0, 0))

img.show()
