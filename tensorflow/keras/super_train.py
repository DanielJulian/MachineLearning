# https://github.com/7AM7/MnistDigit-Recognition
# https://www.facebook.com/ahmed.moorsy.35/posts/666350267079630
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.layers.normalization import BatchNormalization
import keras
from keras.datasets import mnist
from keras import backend as K

# load json and create model
from keras.preprocessing.image import ImageDataGenerator



def save_model(model):
    # serialize model to JSON
    # model_json = model.to_json()
    # with open("model/model.json", "w") as json_file:
    #     json_file.write(model_json)
    # serialize weights to HDF5
    model.save("supermodel.h5")
    print("Saved model to disk")


def build_model():
    model = Sequential()

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        input_shape=(28, 28, 1)))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])

    return model




batch_size = 128
num_classes = 10
epochs = 40

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = build_model()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 16)


datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

score = model.evaluate(x_test, y_test, verbose=0)
save_model(model)
print('Loss:', score[0])
print('Accuracy:', score[1])


'''
# Original:

# https://github.com/7AM7/MnistDigit-Recognition
# https://www.facebook.com/ahmed.moorsy.35/posts/666350267079630
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.layers.normalization import BatchNormalization
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.models import model_from_json


def save_model(model):
    # serialize model to JSON
    # model_json = model.to_json()
    # with open("model/model.json", "w") as json_file:
    #     json_file.write(model_json)
    # serialize weights to HDF5
    model.save("supermodel.h5")
    print("Saved model to disk")


def build_model():
    model = Sequential()

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        input_shape=(28, 28, 1)))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))

    model.add(Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model




batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = build_model()
model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
save_model(model)
print('Loss:', score[0])
print('Accuracy:', score[1])



'''