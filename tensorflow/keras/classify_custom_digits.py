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
    #print(digits_coords)
    return digits_coords

def get_digit(image, coords):
    digit_image = image.crop((coords[0][0], coords[0][1], coords[1][0], coords[1][1]))
    digit_image = digit_image.resize((28, 28))
    #digit_image.show()
    return digit_image


def main():
    # Load pretrained model
    mnist_model = load_model('mnist_model_kaggle.h5')
    #print(mnist_model.summary())

    # Load image with aligned digits.
    original_image = cv2.imread("my_images/many/numbers3.jpg", 0)

    # Apply some blur
    blurred_image = cv2.GaussianBlur(original_image, (3, 3), 0)
    # Binarize image so we only have black and white pixels
    filtered_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 405, 30)
    # Invert pixel values
    bitnot = cv2.bitwise_not(filtered_image)
    # cv2.imshow('image', bitnot)
    # Load the preproccesed image to PIL library
    grayscaled_image = Image.fromarray(bitnot)
    # Get the digits coordinates from the image
    digit_coordinates = get_digits_coords(grayscaled_image)

    draw = ImageDraw.Draw(grayscaled_image)
    for coords in digit_coordinates:
        digit_image = get_digit(grayscaled_image, coords)
        # digit_image.show()
        image_bytes = digit_image.tobytes()
        # image needs to be a 'batch' though only of one, and with one channel -- grayscale
        image_array = np.reshape(np.frombuffer(image_bytes, dtype=np.uint8), (1, 28, 28, 1))

        image_array = image_array / 255
        prediction = mnist_model.predict(image_array)
        # argmax to reverse the one hot encoding
        digit = np.argmax(prediction[0])
        # need to convert to int -- numpy.int64 isn't known to serialize
        print({'digit': int(digit)})
        draw.text((coords[1][0] - 50, 0), str(int(digit)), 255)

    grayscaled_image.show()


main()
