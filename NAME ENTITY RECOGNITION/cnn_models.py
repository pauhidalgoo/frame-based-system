import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout

def create_model(model_type):
    model = Sequential()
    
    if model_type == 'Conv1D_32':
        model.add(Conv1D(32, kernel_size=3, padding="same", activation="relu"))
    elif model_type == 'Conv1D_64':
        model.add(Conv1D(64, kernel_size=3, padding="same", activation="relu"))
    elif model_type == 'Conv1D_128':
        model.add(Conv1D(128, kernel_size=3, padding="same", activation="relu"))
    elif model_type == 'Conv1D_256':
        model.add(Conv1D(256, kernel_size=3, padding="same", activation="relu"))

    return model

model_types = [
    'Conv1D_32',
    'Conv1D_64',
    'Conv1D_128',
    'Conv1D_256',]