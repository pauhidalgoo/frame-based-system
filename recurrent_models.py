import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout

def create_model(model_type):
    model = Sequential()
    
    if model_type == 'SimpleRNN':
        model.add(SimpleRNN(128, return_sequences=False))
    elif model_type == 'SimpleRNN_Max':
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'SimpleRNN_Max64':
        model.add(SimpleRNN(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'SimpleRNN_Max256':
        model.add(SimpleRNN(256, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'SimpleRNN_Avg64':
        model.add(SimpleRNN(64, return_sequences=True))
        model.add(GlobalAveragePooling1D())
    elif model_type == 'SimpleRNN_Avg256':
        model.add(SimpleRNN(256, return_sequences=True))
        model.add(GlobalAveragePooling1D())
    elif model_type == 'LSTM':
        model.add(LSTM(128, return_sequences=False))
    elif model_type == 'LSTM_16':
        model.add(LSTM(16, return_sequences=False))
    elif model_type == 'LSTM_32':
        model.add(LSTM(32, return_sequences=False))
    elif model_type == 'LSTM_Max':
        model.add(LSTM(128, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'LSTM_Max_16':
        model.add(LSTM(16, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'LSTM_Max_32':
        model.add(LSTM(32, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'LSTM_Max_64':
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'LSTM_Max_256':
        model.add(LSTM(256, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'LSTM_Average':
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalAveragePooling1D())
    elif model_type == 'LSTM_Average_128':
        model.add(LSTM(128, return_sequences=True))
        model.add(GlobalAveragePooling1D())
    elif model_type == 'GRU_Max':
        model.add(GRU(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'GRU_Max_32':
        model.add(GRU(32, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'GRU_Average':
        model.add(GRU(64, return_sequences=True))
        model.add(GlobalAveragePooling1D())
    elif model_type == 'BidirectionalLSTM':
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
    elif model_type == 'BidirectionalLSTM_Max':
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalLSTM_Max_ave':
        model.add(Bidirectional(LSTM(64, return_sequences=True), merge_mode="ave"))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalLSTM_Max_sum':
        model.add(Bidirectional(LSTM(64, return_sequences=True), merge_mode="sum"))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'ConvLSTM':
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(LSTM(64, return_sequences=False))
    elif model_type == 'ConvLSTM_32':
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(LSTM(32, return_sequences=False))
    elif model_type == 'ConvLSTM_Max':
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'ConvLSTM_Max_32':
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(LSTM(32, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'Conv128LSTM_Max':
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'Conv128LSTM_Max_128':
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(LSTM(128, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'ConvConvLSTM_Max':
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'ConvConvLSTM_Max_v2':
        model.add(Conv1D(16, kernel_size=5, activation='relu'))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'ConvGRU':
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(GRU(64, return_sequences=False))
    elif model_type == 'ConvGRU_Max':
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(GRU(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'StackedLSTM':
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32, return_sequences=False))
    elif model_type == 'StackedLSTM_Max':
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'StackedGRU':
        model.add(GRU(64, return_sequences=True))
        model.add(GRU(32, return_sequences=False))
    elif model_type == 'StackedGRU_Max':
        model.add(GRU(64, return_sequences=True))
        model.add(GRU(32, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalGRU':
        model.add(Bidirectional(GRU(128, return_sequences=False)))
    elif model_type == 'BidirectionalGRU_Max':
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalStackedLSTM':
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(LSTM(64, return_sequences=False))
    elif model_type == 'BidirectionalStackedLSTM_Max':
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(LSTM(16, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalStackedLSTM_Max_Medium':
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(LSTM(32, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalStackedLSTM_Max_Big':
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(LSTM(64, return_sequences=True))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalConvLSTM':
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(Bidirectional(LSTM(128, return_sequences=False)))
    elif model_type == 'BidirectionalConvLSTM_Max':
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'DropoutLSTM':
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'DropoutGRU':
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'DropoutLSTM_128':
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'DropoutGRU_128':
        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalStackedLSTMLSTM_Max':
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalStackedGRUGRU_Max':
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Bidirectional(GRU(32, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalDropoutLSTM':
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'BidirectionalDropoutGRU':
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
    
    return model


model_types = [
     'BidirectionalStackedLSTMLSTM_Max', 'BidirectionalStackedGRUGRU_Max',
    'BidirectionalDropoutLSTM', 'BidirectionalDropoutGRU', 'BidirectionalLSTM_Max_ave',
    'BidirectionalLSTM_Max_sum'
]


"""
'SimpleRNN', 'SimpleRNN_Max', 'SimpleRNN_Max64', 'SimpleRNN_Max256',
    'SimpleRNN_Avg64', 'SimpleRNN_Avg256', 'LSTM', 'LSTM_16', 'LSTM_32',
    'LSTM_Max', 'LSTM_Max_16', 'LSTM_Max_32', 'LSTM_Max_64', 'LSTM_Max_256',
    'LSTM_Average', 'LSTM_Average_128', 'GRU_Max', 'GRU_Max_32', 'GRU_Average',
    'BidirectionalLSTM', 'BidirectionalLSTM_Max', 'ConvLSTM', 'ConvLSTM_32',
    'ConvLSTM_Max', 'ConvLSTM_Max_32', 'Conv128LSTM_Max', 'Conv128LSTM_Max_128',
    'ConvConvLSTM_Max', 'ConvConvLSTM_Max_v2', 'ConvGRU', 'ConvGRU_Max',
    'StackedLSTM', 'StackedLSTM_Max', 'StackedGRU', 'StackedGRU_Max',
    'BidirectionalGRU', 'BidirectionalGRU_Max', 'BidirectionalStackedLSTM',
    'BidirectionalStackedLSTM_Max', 'BidirectionalStackedLSTM_Max_Medium',
    'BidirectionalStackedLSTM_Max_Big', 'BidirectionalConvLSTM',
    'BidirectionalConvLSTM_Max', 'DropoutLSTM', 'DropoutGRU', 'DropoutLSTM_128',
    'DropoutGRU_128',
"""