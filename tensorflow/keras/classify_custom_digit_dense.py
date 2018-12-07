import numpy as np
from keras.models import load_model
from PIL import Image
from PIL.ImageOps import fit, grayscale

'''
def interpolate(value, origmin, origmax, targetmin, targetmax):
    # Figure out how 'wide' each range is
    leftSpan = origmax - origmin
    rightSpan = targetmax - targetmin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - origmin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return targetmin + (valueScaled * rightSpan)
    '''

MNIST_MODEL = load_model('network_2.h5')
print(MNIST_MODEL.summary())


#using Pillow -- python image processing -- to turn the poseted file into bytes
image = Image.open('my_images/individual/preprocessed/uno2828.jpg')
# image = grayscale(image.resize((28, 28)))
image_bytes = image.tobytes()
#image needs to be a 'batch' though only of one, and with one channel -- grayscale
image_array = np.reshape(np.frombuffer(image_bytes,  dtype=np.uint8), (1, 784))

image_array = image_array/255
print(image_array)
prediction = MNIST_MODEL.predict(image_array)
#argmax to reverse the one hot encoding
digit = np.argmax(prediction[0])
#need to convert to int -- numpy.int64 isn't known to serialize
print({'digit': int(digit)})
