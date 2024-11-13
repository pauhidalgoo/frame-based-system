import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout

def create_model(model_type):
    model = Sequential()
    

    if model_type == 'BidirectionalGRU_128_01':
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Dropout(0.1))
    elif model_type == 'BidirectionalGRU_256_01':
        model.add(Bidirectional(GRU(256, return_sequences=True)))
        model.add(Dropout(0.1))
    elif model_type == 'BidirectionalGRU_256':
        model.add(Bidirectional(GRU(256, return_sequences=True)))

    return model
model_types = [
    'BidirectionalGRU_128_01',
    'BidirectionalGRU_256_01',
    'BidirectionalGRU_256'
]
