import os
import matplotlib.pyplot as plt
import itertools
import operator
import cv2
from neural_network import predict_image
from preprocess import preprocessing
from segmentation import segmentation


# El fondo de la imagen de debe ser lo mas claro posible, y los digitos lo mas obscuros posibles.

def perform_ocr(raw_image, model_dict):

    # Borro todo el contenido de la carpeta de output
    fileList = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/output_images")
    for fileName in fileList:
        os.remove(os.path.dirname(os.path.abspath(__file__)) + "/output_images/" + fileName)

    # Redimensionamos la region de interes seleccionada a 100 px de altura. La anchura dependerá de la resolución original.
    height = raw_image.shape[0]
    width = raw_image.shape[1]
    aspectRatio = width / (height * 1.0)
    height = 100
    width = int(height * aspectRatio)
    raw_image = cv2.resize(raw_image, (width, height))

    # Aplico un procesamiento a la imagen
    preprocessed_image = preprocessing(raw_image)

    # Segmento y redimensiono los dígitos para mantener la relación de aspecto
    all_digits = segmentation(preprocessed_image)
    all_results = ["" for _ in range(0, len(model_dict))]
    best_result = ""
    for digit in all_digits:
        current_predicted_digit = []
        '''
        plt.imshow(digit)
        plt.show()
        '''
        for key, value in model_dict.items():
            predicted = predict_image(digit, value['model'], value['graph'], value['session'])
            all_results[key] += str(predicted)
            current_predicted_digit.append(str(predicted))
        # De todas las predicciones, supongo que el digito correcto es el que más veces se predijo
        best_result += get_best_digit(current_predicted_digit)
    print(all_results)
    print(best_result)
    return best_result


def get_best_digit(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]
