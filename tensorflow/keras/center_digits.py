import numpy as np
import cv2
from keras.models import load_model
from PIL import Image, ImageDraw


# Retorna lista de listas, donde cada lista tiene 2 tuplas, la primera indicando el inicio y las segunda el final del
# digito.
def get_digits_coords(inverted_image):
    digits_coords = []
    pixels = inverted_image.load()
    width, height = inverted_image.size

    sumaAnterior = 0
    for i in range(0, width):
        sumaActual = 0
        for j in range(0, height):
            sumaActual += pixels[i, j]
        if sumaAnterior == 0 and sumaActual > 0:
            inicio = i
        elif (sumaAnterior > 0 and sumaActual == 0) or (i == width and sumaActual > 0):
            digits_coords.append([(inicio, 0), (i, height)])
        sumaAnterior = sumaActual
    print("Digits Found: " + str(len(digits_coords)))
    return digits_coords


def chop_black_rows(digit_coordinates, image):
    digits_coords = []
    for coords in digit_coordinates:
        digit_image = image.crop((coords[0][0], coords[0][1], coords[1][0], coords[1][1]))
        pixels = digit_image.load()
        width, height = digit_image.size

        first = False
        for y in range(0, height):
            sumaActual = 0
            for x in range(0, width):
                sumaActual += pixels[x, y]
            if not first and sumaActual > 0:
                inicio = y
                first = True
            elif (first and sumaActual == 0) or (y == height and first):
                digits_coords.append([(coords[0][0], inicio), (coords[1][0], y)])
                #current = [(coords[0][0], inicio), (coords[1][0], y)]
                break

        #temp = image.crop((current[0][0], current[0][1], current[1][0], current[1][1]))
        #temp.show()
    return digits_coords


def get_digit(image, coords):
    digit_image = image.crop((coords[0][0], coords[0][1], coords[1][0], coords[1][1]))
    digit_image = digit_image.resize((28, 28))
    #digit_image.show()
    return digit_image


def black_bg_square(img):
    background = Image.new('L', (38, 38), 0)
    offset = (5, 5)
    background.paste(img, offset)
    background = background.resize((28, 28))
    return background


def filter_wrong_digits(digit_coordinates):
    filtered_coordinates = []
    # First i get the median width
    sum = 0
    for coords in digit_coordinates:
        sum += int(coords[1][0] - coords[0][0])
    median_length = sum/len(digit_coordinates)
    for coords in digit_coordinates:
        if not int(coords[1][0] - coords[0][0]) < (median_length/2):
            filtered_coordinates.append(coords)

    print("New coordinates filtered: " + str(len(filtered_coordinates)))
    return filtered_coordinates


def main():
    # Load pretrained model
    mnist_model = load_model('mnist_model_kaggle_2.h5')
    # print(mnist_model.summary())

    # Load image with aligned digits.
    original_image = cv2.imread("my_images/many/unos.jpg", 0)

    # Apply some blur
    blurred_image = cv2.GaussianBlur(original_image, (3, 3), 0)
    # Binarize image so we only have black and white pixels
    filtered_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 405,
                                           30)
    # Invert pixel values
    bitnot = cv2.bitwise_not(filtered_image)
    # cv2.imshow('image', bitnot)
    # Load the preproccesed image to PIL library
    grayscaled_image = Image.fromarray(bitnot)
    # Get the digits coordinates from the image
    digit_coordinates = get_digits_coords(grayscaled_image)
    digit_coordinates = chop_black_rows(digit_coordinates, grayscaled_image)
    digit_coordinates = filter_wrong_digits(digit_coordinates)
    draw = ImageDraw.Draw(grayscaled_image)
    for coords in digit_coordinates:
        digit_image = get_digit(grayscaled_image, coords)
        digit_image = black_bg_square(digit_image)
        digit_image.show()
        image_bytes = digit_image.tobytes()
        # image needs to be a 'batch' though only of one, and with one channel -- grayscale
        image_array = np.reshape(np.frombuffer(image_bytes, dtype=np.uint8), (1, 28, 28, 1))

        image_array = image_array / 255
        prediction = mnist_model.predict(image_array)
        # argmax to reverse the one hot encoding
        digit = np.argmax(prediction[0])
        # need to convert to int -- numpy.int64 isn't known to serialize
        print({'digit': int(digit)})
        draw.text((coords[1][0] - 50, coords[0][1] - 15), str(int(digit)), 255)

    grayscaled_image.show()


main()
