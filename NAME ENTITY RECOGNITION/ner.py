import random
import pandas as pd
import csv
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from datetime import datetime
from collections import Counter

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D, Dropout, Conv1D, GlobalAveragePooling1D,  SimpleRNN, LSTM, GRU, Bidirectional, TimeDistributed
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from keras.metrics import F1Score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K


class NamedEntityRecognition:
    """
    A class for training and evaluating a NER model using Keras and TensorFlow.

    Attributes:
        hyperparams (dict): A dictionary of hyperparameters for model training.
            vocab_size
            embedding_dim
            epochs
            batch_size
        prep_config (dict): A dictionary of parameters for sentence preprocessing
            lemmatize
            stem
            remove_stopwords
            custom_stopwords
        train_config (dict): A dictionary of parameters for training
            selection_metric
            f1_type
            use_sample_weights
        model (keras.models.Sequential): A Keras Sequential model to be trained.
    """
    def __init__(self, model, hyperparams = {}, prep_config = {}, train_config = {}, training_times=1, automatic_train=False, verbosing=0, name=f"test_{datetime.now().strftime('%m%d_%H%M')}", use_augmented_data=False, save_results=True, results_file_name = "./results/NER_results"):
        """
        Initializes the NER class with hyperparameters and a Keras model.

        Args:
            hyperparams (dict): A dictionary containing hyperparameters like 'vocab_size', 'embedding_dim', 'epochs', and 'batch_size'.
            model (keras.models.Sequential): A Keras Sequential model to be trained.
            training_times (int): Number of times the model should be trained.
            automatic_train (bool): If True, automatically train and evaluate the model upon initialization.
        """
        self.architecture_name = name
        self.save_results = save_results
        self.results_file = results_file_name
        default_hyperparams = {'vocab_size': 500, 'embedding_dim': 768, 'epochs': 5, 'batch_size': 32}
        self.hyperparams = {**default_hyperparams, **hyperparams}
        default_config = {'lemmatize':False, 'stem':False, 'remove_stopwords':False, 'custom_stopwords':None, 'padding':'pre'}
        self.prep_config = {**default_config, **prep_config}
        default_train = {'selection_metric':"accuracy", 'f1_type':"macro", 'use_sample_weights':True, 'early_stopping': True, 'early_stopping_patience': 5}
        self.train_config = {**default_train, **train_config}
        self.initial_model = model
        self.training_times = training_times
        self.verbosing = verbosing
        self.use_augmented_data = use_augmented_data
        self.training_information = {}
        self._load_data()
        self.preprocess_data()

        if automatic_train:
            self.train_model()
            self.evaluate_model()
    
    def _load_data(self):
        """
        Loads the training, validation, and test datasets.
        """
        train_data = pd.read_csv('../data/train.csv', header=None)
        test_data = pd.read_csv('../data/test.csv', header=None)
        val_data = train_data.tail(900)
        if self.use_augmented_data:
            train_data = pd.read_csv('../train_data_augmented.csv', header=None)
        else:
            train_data = pd.read_csv('../data/train.csv', header=None, nrows=4078)
        
        
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    
    def _format_labels(self, labels):
        """
        Formats the labels by removing unwanted characters and spaces.

        Args:
            labels (list): A list of label strings.

        Returns:
            list: A list of formatted label strings.
        """
        labels = list(s.replace('"', '') for s in labels)
        return labels
    
    def _remove_sentences(self, list_labels, list_sequences):
        """
        Removes specified values and their corresponding indices from labels and sequences.

        Args:
            values_to_remove (list): A list of label values to remove.
            labels_list (list): The list of labels.
            pad_sequences_list (list): The list of padded sequences.

        Returns:
            tuple: A tuple containing the cleaned labels list and the cleaned padded sequences array.
        """
        idx_to_remove = []
        labels_to_remove = []
        for idx, labels in enumerate(list_labels):
            for label in labels:
                if label not in self.unique_entities:
                    idx_to_remove.append(idx)
                    labels_to_remove.append(label)

        labels = [elem for i, elem in enumerate(list_labels) if i not in idx_to_remove]
        sequences = [elem for i, elem in enumerate(list_sequences) if i not in idx_to_remove]
        return labels, np.array(sequences)
    
    def _save_results(self, results_dict, file_path, header=None):
        if self.save_results != None:
            if self.save_results == "few" and file_path == self.results_file + "_complete.csv":
                pass
            with open(file_path, mode='a', newline='', encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=header if header else results_dict[0].keys())
                # writer.writeheader() # Uncomment if the files aren't created
                writer.writerows(results_dict)

    def _get_model_summary(self, model: tf.keras.Model) -> str:
        
        string_list = []
        model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
        string_list = [str(a) for a in string_list]
        return "\n".join(string_list)
    
    def _count_unique_entities(self, list_of_label_sentences):
        flat_labels = []
        for labels in list_of_label_sentences:
            flat_labels += labels.split()
        c = Counter(flat_labels)
        return len(c), list(c.keys())

    def preprocess_data(self):
        """
        Preprocesses the data by tokenizing sentences, sequencing, padding, and encoding labels.
        """
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        if self.prep_config['custom_stopwords']:
            stop_words.update(self.prep_config['custom_stopwords'])

        def preprocess_text(sentence):
            words = word_tokenize(sentence)
            if self.prep_config['lemmatize']:
                words = [lemmatizer.lemmatize(word) for word in words]
            if self.prep_config['stem']:
                words = [stemmer.stem(word) for word in words]
            return ' '.join(words)

        # Training data
        self.train_sentences = [preprocess_text(sent) for sent in list(self.train_data[0])]
        self.train_labels = self._format_labels(list(self.train_data[1]))

        num_unique_entities, self.unique_entities = self._count_unique_entities(self.train_labels)


        # Tokenize the sentences
        self.tokenizer = Tokenizer(self.hyperparams['vocab_size'])
        self.tokenizer.fit_on_texts(self.train_sentences)

        # Sequence the sentences
        self.train_sequences = self.tokenizer.texts_to_sequences(self.train_sentences)

        max_seq_len = max(map(len, self.train_sequences))
        self.max_seq_len = max_seq_len
        self.train_pad_sequences = pad_sequences(self.train_sequences, maxlen=max_seq_len, padding=self.prep_config['padding'])

        # Encode the labels
        label_encoder = LabelEncoder()
        self.label_encoder = label_encoder.fit(["<pad>"]+list(self.unique_entities))
        label_encoder = self.label_encoder
        self.train_numerical_labels =[label_encoder.transform(t.split()) for t in self.train_labels]
        self.train_pad_labels = pad_sequences(self.train_numerical_labels, maxlen = max_seq_len, padding=self.prep_config['padding'])

        num_classes = len(self.unique_entities)
        self.train_labels_one_hot = [to_categorical(a, num_classes +1) for a in self.train_pad_labels]

        # Validation and test data
        self.val_labels = self._format_labels(list(self.val_data[1]))
        self.test_labels = self._format_labels(list(self.test_data[1]))


        self.val_sentences = [preprocess_text(sent) for sent in list(self.val_data[0])]
        self.test_sentences = [preprocess_text(sent) for sent in list(self.test_data[0])]

        self.val_sequences = self.tokenizer.texts_to_sequences(self.val_sentences)
        self.test_sequences = self.tokenizer.texts_to_sequences(self.test_sentences)

        val_pad_sequences_old = pad_sequences(self.val_sequences, maxlen=max_seq_len, padding=self.prep_config['padding'])
        test_pad_sequences_old = pad_sequences(self.test_sequences, maxlen=max_seq_len, padding=self.prep_config['padding'])


        _test_labels = [label.split() for label in self.test_labels]
        _val_labels = [label.split() for label in self.val_labels]
        val_labels2, self.val_pad_sequences = self._remove_sentences(_val_labels, val_pad_sequences_old)
        test_labels2, self.test_pad_sequences = self._remove_sentences(_test_labels, test_pad_sequences_old)

        val_numerical_labels = [self.label_encoder.transform(t) for t in val_labels2]
        test_numerical_labels = [self.label_encoder.transform(t) for t in test_labels2]


        self.val_pad_labels = pad_sequences(val_numerical_labels, maxlen=max_seq_len, padding=self.prep_config['padding'])
        self.test_pad_labels = pad_sequences(test_numerical_labels, maxlen=max_seq_len, padding=self.prep_config['padding'])



        self.val_labels_one_hot = [to_categorical(a, num_classes +1) for a in self.val_pad_labels]
        self.test_labels_one_hot = [to_categorical(a, num_classes +1) for a in self.test_pad_labels]
        self.num_classes = num_classes

        # Print number of classes of training val and test
        if self.verbosing:
            print(f"Number of classes in training data: {num_classes}")

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_pad_labels.flatten()),
            y=self.train_pad_labels.flatten()
        )

        class_weights_dict = dict(enumerate(class_weights))

        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

        self.sample_weights = np.ones(self.train_pad_labels.shape)
        for class_label, weight in class_weights_dict.items():
            self.sample_weights[self.train_pad_labels == class_label] = weight

    @staticmethod
    def f1(y_true, y_pred):
        # Fet amb el chat nidea de si esta be
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
    def train_model(self):
        """
        Trains the Keras model multiple times using the training data and hyperparameters.
        Calculates average metrics and identifies the best model based on validation accuracy.
        Stores the information in self.training_information.
        """
        if not isinstance(self.initial_model, Sequential):
            raise ValueError("Model must be an instance of a Keras Sequential model")
        

        sample_weights = self.sample_weights if self.train_config['use_sample_weights'] else None
        # Extract layers from the initial model
        initial_layers = self.initial_model.layers
        # Vocabulary size and embedding dimensions
        vocab_size = self.hyperparams['vocab_size'] + 1
        embedding_dim = self.hyperparams['embedding_dim']

        # Initialize lists to collect metrics across all training runs
        training_acc_list = []
        training_f1_list = []
        training_loss_list = []
        val_acc_list = []
        val_f1_list = []
        val_loss_list = []
        histories = []

        # Variables to track the best model
        best_val_acc = -np.inf
        best_val_f1 = -np.inf
        best_history = None
        best_model = None
        complete_results = []

        for i in range(self.training_times):
            print(f"\rTraining model {i+1}/{self.training_times}", end='', flush=True)

            # Rebuild the model for each training run
            self.model = Sequential()
            self.model.add(Embedding(vocab_size, embedding_dim))  # Embedding layer
            for layer in initial_layers:
                # Clone each layer to ensure independence between models
                config = layer.get_config()
                cloned_layer = layer.__class__.from_config(config)
                self.model.add(cloned_layer)
            self.model.add(TimeDistributed(Dense(self.num_classes + 1, activation="softmax")))  # Output layer

            # Compile the model
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', self.f1])

            callbacks_list = []
            if self.train_config['early_stopping']:
                callbacks_list.append(EarlyStopping(patience=self.train_config['early_stopping_patience']))
            # Fit the model and capture the training history
            history = self.model.fit(
                self.train_pad_sequences, 
                np.array(self.train_labels_one_hot),
                batch_size=self.hyperparams['batch_size'], 
                epochs=self.hyperparams['epochs'], 
                validation_data=(self.val_pad_sequences, np.array(self.val_labels_one_hot)),
                sample_weight = sample_weights,
                callbacks = callbacks_list,
                verbose=self.verbosing
            )

            histories.append(history.history)

            for epoch in range(len(history.history['accuracy'])):
                epoch_result = {
                    'architecture_name': self.architecture_name,
                    'summary': self.model.get_config(),
                    'run_number': i + 1,
                    'epoch': epoch + 1,
                    'training_acc': history.history['accuracy'][epoch],
                    'training_f1': history.history['f1'][epoch],
                    'training_loss': history.history['loss'][epoch],
                    'val_acc': history.history['val_accuracy'][epoch],
                    'val_f1': history.history['val_f1'][epoch],
                    'val_loss': history.history['val_loss'][epoch],
                    **self.hyperparams, # ** el que fa és separa les keys i values dels diccionaris ;) tope útil ho vaig aprendre fa poc
                    **self.prep_config, # un sol * crec que separa els valors d'un iterador
                    **self.train_config
                }
                complete_results.append(epoch_result)

            # Extract the final epoch's metrics
            final_epoch = len(history.history['accuracy']) - 1
            training_acc = history.history['accuracy'][final_epoch]
            training_f1 = history.history['f1'][final_epoch]

            training_loss = history.history['loss'][final_epoch]
            val_acc = history.history['val_accuracy'][final_epoch]
            val_f1 = history.history['val_f1'][final_epoch]
            val_loss = history.history['val_loss'][final_epoch]

            # Append metrics to the respective lists
            training_acc_list.append(training_acc)
            training_f1_list.append(training_f1)
            training_loss_list.append(training_loss)
            val_acc_list.append(val_acc)
            val_f1_list.append(val_f1)
            val_loss_list.append(val_loss)

            # Update the best model if current model has higher validation accuracy
            if self.train_config['selection_metric'] == "accuracy":
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_history = history.history
                    
                    # Copy the best model
                    if not self.architecture_name.startswith("Bidirectional"):
                        best_model = tf.keras.models.clone_model(self.model)
                        best_model.set_weights(self.model.get_weights())
            else:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_history = history.history
                    
                    # Copy the best model
                    if not self.architecture_name.startswith("Bidirectional"):
                        best_model = tf.keras.models.clone_model(self.model)
                        best_model.set_weights(self.model.get_weights())

        # Calculate average metrics across all training runs
        average_training_acc = np.mean(training_acc_list)
        average_training_f1 = np.mean(training_f1_list)
        average_training_loss = np.mean(training_loss_list)
        average_val_acc = np.mean(val_acc_list)
        average_val_f1 = np.mean(val_f1_list)
        average_val_loss = np.mean(val_loss_list)

        # Extract the best model's per-epoch training and validation accuracy
        best_model_training_acc = best_history['accuracy']
        best_model_validation_acc = best_history['val_accuracy']

        best_model_training_f1 = best_history['f1']
        best_model_validation_f1 = best_history['val_f1']

        # Set the self.model to the best model
        if not self.architecture_name.startswith("Bidirectional"):
            self.model = tf.keras.models.clone_model(best_model)
            self.model.set_weights(best_model.get_weights())
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', self.f1])

        # Store all the collected information in the training_information dictionary
        self.training_information = {
            'average_training_acc': average_training_acc,
            'average_training_f1': average_training_f1,
            'average_training_loss': average_training_loss,
            'average_val_acc': average_val_acc,
            'average_val_f1': average_val_f1,
            'average_val_loss': average_val_loss,
            'best_model_training_acc': best_model_training_acc,
            'best_model_training_f1': best_model_training_f1,
            'best_model_validation_acc': best_model_validation_acc,
            'best_model_validation_f1': best_model_validation_f1
        }

        self._save_results(complete_results, self.results_file + '_complete.csv')
        average_metrics = [{
            'architecture_name': self.architecture_name,
            'summary': self.model.get_config(),
            'average_training_acc': average_training_acc,
            'average_training_f1': average_training_f1,
            'average_training_loss': average_training_loss,
            'average_val_acc': average_val_acc,
            'average_val_f1': average_val_f1,
            'average_val_loss': average_val_loss,
            **self.hyperparams,
            **self.prep_config,
            **self.train_config
        }]
        self._save_results(average_metrics, self.results_file + '_average.csv')

        best_metrics = [{
            'architecture_name': self.architecture_name,
            'summary': self.model.get_config(),
            'best_model_training_acc': best_model_training_acc[-1],
            'best_model_training_f1': best_model_training_f1[-1],
            'best_model_validation_acc': best_model_validation_acc[-1],
            'best_model_validation_f1': best_model_validation_f1[-1],
            **self.hyperparams,
            **self.prep_config,
            **self.train_config
        }]
        self._save_results(best_metrics,  self.results_file + '_best.csv')


        # Empty prints for new line
        print('\n')

    
    def print_training_information(self):
        """
        Prints the training information containing average metrics and best model details.
        """
        print("Average Training Accuracy:", self.training_information['average_training_acc'])
        print("Average Training F1:", self.training_information['average_training_f1'])
        print("Average Training Loss:", self.training_information['average_training_loss'])
        print("Average Validation Accuracy:", self.training_information['average_val_acc'])
        print("Average Validation F1:", self.training_information['average_val_f1'])
        print("Average Validation Loss:", self.training_information['average_val_loss'])
        print("Best Model Validation Accuracy:", self.training_information['best_model_validation_acc'][-1])
        print("Best Model Validation F1:", self.training_information['best_model_validation_f1'][-1])
        print()
    
    def evaluate_model(self):
        """
        Evaluates the trained model on the test data and prints the accuracy.
        Note: This evaluates the last trained model.
        """
        print('Evaluating model...')
        loss, accuracy, f1 = self.model.evaluate(
            self.test_pad_sequences, 
            self.test_encoded_labels, 
            batch_size=self.hyperparams['batch_size']
        )

        f1 = np.mean(f1)
        print(f"Test accuracy: {accuracy}")
        print(f"Test Macro F1: {f1}")
    
    def get_training_information(self):
        """
        Retrieves the training information containing average metrics and best model details.

        Returns:
            dict: A dictionary containing average training/validation metrics and best model's metrics.
        """
        return self.training_information
    
    def view_wrong_predictions(self):
        probs = self.model.predict(self.test_pad_sequences)
        _predicted_labels = np.argmax(probs, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(_predicted_labels)

        for i in range(0, len(predicted_labels)):
            if self.test_labels[i] != predicted_labels[i]:
                print(i)
                print('Sentence: ', self.test_sentences_removed[i])
                print('Original label: ', self.test_labels[i])
                print('Predicted label: ', predicted_labels[i])
                print()