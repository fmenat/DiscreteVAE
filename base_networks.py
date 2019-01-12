import keras
from keras.layers import *
from keras.models import Sequential,Model
from keras import backend as K


def define_pre_encoder(data_dim,layers=2,units=512,dropout=0.0,BN=False): #define pre_encoder network
    model = Sequential(name='pre-encoder')
    model.add(InputLayer(input_shape=(data_dim,)))
    for i in range(1,layers+1):
        #model.add(Dense(int(units/i), activation='relu'))
        model.add(Dense(units,activation='relu'))
        if dropout != 0. and dropout != None:
            model.add(Dropout(dropout))
        if BN:
            model.add(BatchNormalization())
    return model

def define_generator(Nb,data_dim,layers=2,units=32,dropout=0.0,BN=False,exclusive=True):
    model = Sequential(name='generator/decoder')
    model.add(InputLayer(input_shape=(Nb,)))
    for i in np.arange(layers,0,-1):
        #model.add(Dense(int(units/i), activation='relu'))
        model.add(Dense(units,activation='relu'))
        if dropout != 0. and dropout != None:
            model.add(Dropout(dropout))
        if BN:
            model.add(BatchNormalization())
    if exclusive:
        model.add(Dense(data_dim, activation='softmax')) #softmax generator
    else:
        model.add(Dense(data_dim, activation='sigmoid'))
    return model
