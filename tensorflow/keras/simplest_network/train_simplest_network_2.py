import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras.callbacks import TensorBoard


def interpolate(value, origmin, origmax, targetmin, targetmax):
    # Figure out how 'wide' each range is
    leftSpan = origmax - origmin
    rightSpan = targetmax - targetmin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - origmin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return targetmin + (valueScaled * rightSpan)

# Utilizo los estilos de ggplot para los gráficos
plt.style.use('ggplot')
# Para reproducir los resultados en múltiples ejecuciones.
np.random.seed(1000)

# Importo el dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Observamos una imagen de ejemplo
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
#plt.show()
# Observamos la forma de l
print(X_train[0])

# Aplanamos las imagenes. De 28*28 -> 784
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Reducimos los pixeles del rango 0-255 al rango 0-1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

for index, sample in enumerate(X_train):
    for pixel in range(len(X_train[index])):
        X_train[index][pixel] = interpolate(X_train[index][pixel], 0, 255, -1, 1)
for index, sample in enumerate(X_test):
    for pixel in range(len(X_test[index])):
        X_train[index][pixel] = interpolate(X_test[index][pixel], 0, 255, -1, 1)


# Observamos que todo este en orden
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

batch_size = 100  # Numero de imagenes usadas en cada paso de optimización
num_classes = 10  # Una clase por dígito
num_epoch = 40  # Cantidad de veces que se recorre el training set para entrenar la red neuronal

# Convierto los vectores de clases en matrices de clases binarias (One-Hot vectors)
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

print(y_train)
print("Y_train: " + str(Y_train))

# Inicializo los weights y biases
initializer = RandomNormal(mean=0.0, stddev=0.1, seed=None)

# Definimos la arquitectura del modelo
model = Sequential()
model.add(Dense(100, input_shape=(784,), kernel_initializer=initializer, bias_initializer=initializer))
model.add(Activation('tanh'))
model.add(Dense(10, kernel_initializer=initializer, bias_initializer=initializer))  # Ultima layer con una neurona de salida por clase.
model.add(Activation('softmax'))

sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

tb_callback = TensorBoard(log_dir='/logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# Iniciamos el aprendizaje
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=num_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    shuffle=True,
                    callbacks=[tb_callback])

# Graficamos las curvas de loss y la accuracy para los sets de training y validación.
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()

# Evaluamos con el test set que tan bien predice la red neuronal ya entrenada.
score = model.evaluate(X_test, Y_test, verbose=0)
print('test score', score[0])
print('Test accuracy:', score[1])

model.save('mnist_model_dense_2.h5')
