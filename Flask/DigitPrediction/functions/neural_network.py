import numpy as np


def predict_image(image, model, graph, session):
    image_array = np.reshape(image, (1, 28, 28, 1))
    with graph.as_default():
        with session.as_default():
            prediction = model.predict(image_array)
            # Prediction es una lista, donde cada indice es la probabilidad de que sea ese numero.
            # Uso argmax para tomar el indice mayor
            digit = np.argmax(prediction[0])
            return int(digit)
