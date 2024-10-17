import random
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D, Dropout, Conv1D, GlobalAveragePooling1D, LayerNormalization #Remove
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



preprocessing = {'vocab_size': 500, ''}


class IntentRecognition:
    def __init__(self, preprocessing, model):
        self.preprocessing = preprocessing
        self.model = model
        self._load_data()
    
    def _load_data(self):
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')
        val_data = pd.read_csv('data/val.csv')
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def preprocess_data(self):
        self.train_sentences = list(self.train_data[0])
        self.train_labels = list(s.replace('"', '') for s in self.train_data[2])
        self.train_labels = list(s.replace(' ', '') for s in self.train_labels)

        # Tokenize the sentences
        self.tokenizer = Tokenizer(self.preprocessing['vocab_size'])
        self.tokenizer.fit_on_texts(self.train_sentences)

        # Sequence the sentences
        self.train_sequences = self.tokenizer.texts_to_sequences(self.train_sentences)

        max_seq_len = max(map(len, self.train_sequences))
        self.train_pad_sequences = pad_sequences(self.train_sequences, maxlen=max_seq_len, padding='post')

        # Encode the labels
        self.label_encoder = LabelEncoder()
        self.train_numerical_labels = self.label_encoder.fit_transform(self.train_labels)
        self.num_classes = len(self.label_encoder.classes_)


    