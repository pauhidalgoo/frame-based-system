import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout

def create_model(model_type):
    model = Sequential()
    
    if model_type == 'LSTM_128_01':
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
    # NOÉS ESTÀ BE EL PRIMER

    elif model_type == 'BidirectionalLSTM_64_01':
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.1))
    elif model_type == 'BidirectionalLSTM_128_01':
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.1))
    elif model_type == 'BidirectionalLSTM_128_02':
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
    elif model_type == 'BidirectionalLSTM_256_01':
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.1))

    elif model_type == 'BidirectionalGRU_64_01':
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Dropout(0.1))
    elif model_type == 'BidirectionalGRU_128_01':
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Dropout(0.1))
    elif model_type == 'BidirectionalGRU_128_02':
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Dropout(0.2))
    elif model_type == 'BidirectionalLSTM_256_01':
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Dropout(0.1))

    return model
model_types = [
    'LSTM_128_01',
    'BidirectionalLSTM_64_01',
    'BidirectionalLSTM_128_01',
    'BidirectionalLSTM_128_02',
    'BidirectionalLSTM_256_01',
    'BidirectionalGRU_64_01',
    'BidirectionalGRU_128_01',
    'BidirectionalGRU_128_02',
    'BidirectionalLSTM_256_01'
]
