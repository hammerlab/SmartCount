import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image


def get_digit_model(input_shape=(32, 32, 3), mode='all'):
    """Build digit classification CNN
    
    See: https://github.com/tohinz/SVHN-Classifier/blob/master/svhn_classifier.py
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))
    
    if mode == 'head':
        # Free all non-dense layers
        for layer in model.layers[:-3]:
            layer.trainable = False
    elif mode == 'tune':
        # Free all non-dense layers and the last conv block
        for layer in model.layers[:-12]:
            layer.trainable = False

    return model