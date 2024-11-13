import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout

def create_model(model_type):
    model = Sequential()
    
    if model_type == 'SimpleRNN_64':
        model.add(SimpleRNN(64, return_sequences=True))
    # NOÉS ESTÀ BE EL PRIMER
    elif model_type == 'SimpleRNN_128':
        model.add(SimpleRNN(128, return_sequences=True))
    elif model_type == 'SimpleRNN_32':
        model.add(SimpleRNN(32, return_sequences=True))
    elif model_type == 'SimpleRNN_256':
        model.add(SimpleRNN(256, return_sequences=True))

    elif model_type == 'LSTM_16':
        model.add(LSTM(16, return_sequences=True))
    elif model_type == 'LSTM_32':
        model.add(LSTM(32, return_sequences=True))
    elif model_type == 'LSTM_64':
        model.add(LSTM(64, return_sequences=True))
    elif model_type == 'LSTM_128':
        model.add(LSTM(128, return_sequences=True))

    elif model_type == 'LSTM_256':
            model.add(LSTM(256, return_sequences=True))

    elif model_type == 'GRU_16':
        model.add(GRU(16, return_sequences=True))
    elif model_type == 'GRU_32':
        model.add(GRU(32, return_sequences=True))
    elif model_type == 'GRU_64':
        model.add(GRU(64, return_sequences=True))
    elif model_type == 'GRU_128':
        model.add(GRU(128, return_sequences=True))
    elif model_type == 'GRU_256':
        model.add(GRU(256, return_sequences=True))

    elif model_type == 'BidirectionalLSTM_16':
        model.add(Bidirectional(LSTM(16, return_sequences=True)))
    elif model_type == 'BidirectionalLSTM_32':
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
    elif model_type == 'BidirectionalLSTM_64':
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
    elif model_type == 'BidirectionalLSTM_128':
        model.add(Bidirectional(LSTM(128, return_sequences=True)))

    elif model_type == 'BidirectionalGRU_64':
        model.add(Bidirectional(GRU(64, return_sequences=True)))
    elif model_type == 'BidirectionalGRU_128':
        model.add(Bidirectional(GRU(128, return_sequences=True)))


    elif model_type == 'BidirectionalLSTM_Max_ave_32':
        model.add(Bidirectional(LSTM(32, return_sequences=True), merge_mode="ave"))
    elif model_type == 'BidirectionalLSTM_Max_sum_32':
        model.add(Bidirectional(LSTM(32, return_sequences=True), merge_mode="sum"))
    elif model_type == 'BidirectionalLSTM_Max_ave_64':
        model.add(Bidirectional(LSTM(64, return_sequences=True), merge_mode="ave"))
    elif model_type == 'BidirectionalLSTM_Max_sum_64':
        model.add(Bidirectional(LSTM(64, return_sequences=True), merge_mode="sum"))

    elif model_type == 'StackedLSTM_32':
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(32, return_sequences=True))
    elif model_type == 'StackedLSTM_64':
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(64, return_sequences=True))

    elif model_type == 'BidirectionalStackedLSTM_32':
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        
    elif model_type == 'BidirectionalDropoutLSTM':
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
    elif model_type == 'BidirectionalDropoutGRU':
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Dropout(0.5))

    elif model_type == 'DropoutGRU':
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.5))
    elif model_type == 'DropoutLSTM_64':
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.5))

    elif model_type == 'DropoutGRU02':
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.2))
    elif model_type == 'DropoutLSTM_6402':
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))

    
    return model

model_types = [
    'SimpleRNN_64',
    'SimpleRNN_128',
    'SimpleRNN_32',
    'SimpleRNN_256',
    'LSTM_16',
    'LSTM_32',
    'LSTM_64',
    'LSTM_128',
    'LSTM_256',
    'GRU_16',
    'GRU_32',
    'GRU_64',
    'GRU_128',
    'GRU_256',
    'BidirectionalLSTM_16',
    'BidirectionalLSTM_32',
    'BidirectionalLSTM_64',
    'BidirectionalLSTM_128',
    'BidirectionalLSTM_Max_ave_32',
    'BidirectionalLSTM_Max_sum_32',
    'BidirectionalLSTM_Max_ave_64',
    'BidirectionalLSTM_Max_sum_64',
    'StackedLSTM_32',
    'StackedLSTM_64',
    'BidirectionalStackedLSTM_32',
    'BidirectionalStackedLSTM_16',
    'BidirectionalDropoutLSTM',
    'BidirectionalDropoutGRU',
    'DropoutGRU',
    'DropoutLSTM_64',
    'DropoutGRU02',
    'DropoutLSTM_6402'
]
