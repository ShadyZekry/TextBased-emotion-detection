from keras.layers import Input, Conv1D, Dense, MaxPooling1D, Flatten, GRU, Dropout, Concatenate, LSTM
from keras.models import Sequential

def build_cnn_model(input_shape : tuple, num_classes: int) -> Sequential:
    model = Sequential(name='cnn')
    model.add(Input(shape=input_shape))
    model.add(Conv1D(100, kernel_size=3))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    # model.add(Dense(100))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

def build_lstm_model(input_shape:tuple, num_classes: int) -> Sequential:
    model = Sequential(name='lstm')
    model.add(Input(shape=input_shape))
    model.add(LSTM(250))
    model.add(Dense(100))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

def build_gru_model(input_shape: tuple, num_classes: int) -> Sequential:
    model = Sequential(name='gru')
    model.add(Input(shape=input_shape))
    model.add(GRU(250))
    model.add(Dense(100))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

def build_cnn_gru_model(input_shape: tuple, num_classes: int) -> Sequential:
    model = Sequential(name='cnn-gru')
    model.add(Input(shape=input_shape))
    model.add(Conv1D(100, kernel_size=4, data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(GRU(100))
    model.add(Dense(100))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model
    
def build_ann_model(input_shape: tuple, num_classes: int) -> Sequential:
    model = Sequential(name='ann')
    model.add(Input(shape=input_shape))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

def build_cnn_lstm_model(input_shape: tuple, num_classes: int) -> Sequential:
    model = Sequential(name='cnn-lstm')
    model.add(Input(shape=input_shape))
    model.add(Conv1D(100, kernel_size=3, data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(LSTM(100))
    model.add(Dense(100))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

