
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,MaxPooling2D,TimeDistributed
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import ConvLSTM2D
import numpy as np
from random import shuffle
from keras.callbacks import ModelCheckpoint
from keras import optimizers


seq = Sequential()

def loadWeights():
    seq.add(ConvLSTM2D(filters=5, kernel_size=(3, 3), input_shape=(29, 256, 256, 3), padding='same', return_sequences=True))
    seq.add(TimeDistributed(MaxPooling2D((2,2),(2,2))))
    seq.add(TimeDistributed(MaxPooling2D((2,2),(2,2))))
    seq.add(ConvLSTM2D(filters=5, kernel_size=(3, 3), input_shape=(29, 256, 256, 3), padding='same', return_sequences=True))
    seq.add(TimeDistributed(MaxPooling2D((2,2),(2,2))))
    seq.add(ConvLSTM2D(filters=5, kernel_size=(3, 3), input_shape=(29, 256, 256, 3), padding='same', return_sequences=True))
    seq.add(TimeDistributed(MaxPooling2D((2,2),(2,2))))
    seq.add(Flatten())
    seq.add(Dense(1024))
    seq.add(Dense(3))
    seq.add(Activation('softmax'))
    seq.compile(loss='categorical_crossentropy', optimizer='ADAM', metrics=['accuracy'])
    seq.load_weights('weights-improvement-15-0.88.hdf5')
    print('i am done')

def prediction(filename):
    data = np.load(filename + '.npy', allow_pickle=True)
    data_frames = list()
    for x in range(29):
        data_zeros = np.zeros((256, 256, 3), dtype=np.int)
        for y, z in zip(data[x][0], data[x][1]):
            data_zeros[z - 1][y - 1] = 255
        data_frames.append(data_zeros)
    x = np.array([data_frames])
    return seq.predict(x)