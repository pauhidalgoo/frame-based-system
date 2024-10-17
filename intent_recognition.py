import random
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D, Dropout, Conv1D, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class IntentRecognition:
    """
    A class for training and evaluating an intent recognition model using Keras and TensorFlow.

    Attributes:
        hyperparams (dict): A dictionary of hyperparameters for model training.
        model (keras.models.Sequential): A Keras Sequential model to be trained.
    """
    def __init__(self, model, hyperparams = {'vocab_size': 500, 'embedding_dim': 768, 'epochs': 1, 'batch_size': 32}, training_times=1, automatic_train=False, verbosing=0):
        """
        Initializes the IntentRecognition class with hyperparameters and a Keras model.

        Args:
            hyperparams (dict): A dictionary containing hyperparameters like 'vocab_size', 'embedding_dim', 'epochs', and 'batch_size'.
            model (keras.models.Sequential): A Keras Sequential model to be trained.
            training_times (int): Number of times the model should be trained.
            automatic_train (bool): If True, automatically train and evaluate the model upon initialization.
        """
        self.hyperparams = hyperparams
        self.initial_model = model
        self.training_times = training_times
        self.verbosing = verbosing
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
        train_data = pd.read_csv('baiges/data/train.csv', header=None)
        test_data = pd.read_csv('baiges/data/test.csv', header=None)
        val_data = train_data.tail(900)
        train_data = pd.read_csv('baiges/data/train.csv', header=None, nrows=4078)
        
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
        labels = list(s.replace(' ', '') for s in labels)
        return labels
    
    def _remove_values_and_indices(self, values_to_remove, labels_list, pad_sequences_list):
        """
        Removes specified values and their corresponding indices from labels and sequences.

        Args:
            values_to_remove (list): A list of label values to remove.
            labels_list (list): The list of labels.
            pad_sequences_list (list): The list of padded sequences.

        Returns:
            tuple: A tuple containing the cleaned labels list and the cleaned padded sequences array.
        """
        indices_to_remove = [idx for idx, item in enumerate(labels_list) if item in values_to_remove]
        cleaned_labels_list = [item for item in labels_list if item not in values_to_remove]
        cleaned_pad_sequences_list = [item for idx, item in enumerate(pad_sequences_list) if idx not in indices_to_remove]
        return cleaned_labels_list, np.array(cleaned_pad_sequences_list)
    
    def preprocess_data(self):
        """
        Preprocesses the data by tokenizing sentences, sequencing, padding, and encoding labels.
        """

        # Training data
        self.train_sentences = list(self.train_data[0])
        self.train_labels = self._format_labels(list(self.train_data[2]))

        # Tokenize the sentences
        self.tokenizer = Tokenizer(self.hyperparams['vocab_size'])
        self.tokenizer.fit_on_texts(self.train_sentences)

        # Sequence the sentences
        self.train_sequences = self.tokenizer.texts_to_sequences(self.train_sentences)

        max_seq_len = max(map(len, self.train_sequences))
        self.train_pad_sequences = pad_sequences(self.train_sequences, maxlen=max_seq_len, padding='post')

        # Encode the labels
        self.label_encoder = LabelEncoder()
        self.train_numerical_labels = self.label_encoder.fit_transform(self.train_labels)
        self.num_classes = len(self.label_encoder.classes_)
        self.train_encoded_labels = to_categorical(self.train_numerical_labels, num_classes=self.num_classes)

        # Validation and test data
        self.val_sentences = list(self.val_data[0])
        self.test_sentences = list(self.test_data[0])

        self.val_sequences = self.tokenizer.texts_to_sequences(self.val_sentences)
        self.test_sequences = self.tokenizer.texts_to_sequences(self.test_sentences)

        self.val_pad_sequences = pad_sequences(self.val_sequences, maxlen=max_seq_len, padding='post')
        self.test_pad_sequences = pad_sequences(self.test_sequences, maxlen=max_seq_len, padding='post')

        self.val_labels = self._format_labels(list(self.val_data[2]))
        self.test_labels = self._format_labels(list(self.test_data[2]))

        # Remove unwanted labels that don't appear in the training data
        values_to_remove = ['day_name','airfare+flight','flight+airline','flight_no+airline']
        self.val_labels, self.val_pad_sequences = self._remove_values_and_indices(values_to_remove, self.val_labels, self.val_pad_sequences)
        self.test_labels, self.test_pad_sequences = self._remove_values_and_indices(values_to_remove, self.test_labels, self.test_pad_sequences)

        # Encode labels
        self.val_encoded_labels = to_categorical(self.label_encoder.transform(self.val_labels), num_classes=self.num_classes)
        self.test_encoded_labels = to_categorical(self.label_encoder.transform(self.test_labels), num_classes=self.num_classes)


        # Print number of classes of training val and test
        if self.verbosing:
            print(f"Number of classes in training data: {self.num_classes}")
    
    def train_model(self):
        """
        Trains the Keras model multiple times using the training data and hyperparameters.
        Calculates average metrics and identifies the best model based on validation accuracy.
        Stores the information in self.training_information.
        """
        if not isinstance(self.initial_model, Sequential):
            raise ValueError("Model must be an instance of a Keras Sequential model")
        
        # Extract layers from the initial model
        initial_layers = self.initial_model.layers
        # Vocabulary size and embedding dimensions
        vocab_size = self.hyperparams['vocab_size'] + 1
        embedding_dim = self.hyperparams['embedding_dim']

        # Initialize lists to collect metrics across all training runs
        training_acc_list = []
        training_loss_list = []
        val_acc_list = []
        val_loss_list = []
        histories = []

        # Variables to track the best model
        best_val_acc = -np.inf
        best_history = None
        best_model = None

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
            self.model.add(Dense(self.num_classes, activation="softmax"))  # Output layer

            # Compile the model
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Fit the model and capture the training history
            history = self.model.fit(
                self.train_pad_sequences, 
                self.train_encoded_labels, 
                batch_size=self.hyperparams['batch_size'], 
                epochs=self.hyperparams['epochs'], 
                validation_data=(self.val_pad_sequences, self.val_encoded_labels),
                verbose=self.verbosing
            )

            histories.append(history.history)

            # Extract the final epoch's metrics
            final_epoch = self.hyperparams['epochs'] - 1
            training_acc = history.history['accuracy'][final_epoch]
            training_loss = history.history['loss'][final_epoch]
            val_acc = history.history['val_accuracy'][final_epoch]
            val_loss = history.history['val_loss'][final_epoch]

            # Append metrics to the respective lists
            training_acc_list.append(training_acc)
            training_loss_list.append(training_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

            # Update the best model if current model has higher validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_history = history.history
                
                # Copy the best model
                best_model = tf.keras.models.clone_model(self.model)
                best_model.set_weights(self.model.get_weights())


        # Calculate average metrics across all training runs
        average_training_acc = np.mean(training_acc_list)
        average_training_loss = np.mean(training_loss_list)
        average_val_acc = np.mean(val_acc_list)
        average_val_loss = np.mean(val_loss_list)

        # Extract the best model's per-epoch training and validation accuracy
        best_model_training_acc = best_history['accuracy']
        best_model_validation_acc = best_history['val_accuracy']

        # Set the self.model to the best model
        self.model = tf.keras.models.clone_model(best_model)
        self.model.set_weights(best_model.get_weights())
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Store all the collected information in the training_information dictionary
        self.training_information = {
            'average_training_acc': average_training_acc,
            'average_training_loss': average_training_loss,
            'average_val_acc': average_val_acc,
            'average_val_loss': average_val_loss,
            'best_model_training_acc': best_model_training_acc,
            'best_model_validation_acc': best_model_validation_acc
        }

        # Empty prints for new line
        print('\n')

    
    def print_training_information(self):
        """
        Prints the training information containing average metrics and best model details.
        """
        print("Average Training Accuracy:", self.training_information['average_training_acc'])
        print("Average Training Loss:", self.training_information['average_training_loss'])
        print("Average Validation Accuracy:", self.training_information['average_val_acc'])
        print("Average Validation Loss:", self.training_information['average_val_loss'])
        print("Best Model Validation Accuracy:", self.training_information['best_model_validation_acc'][-1])
        print()
    
    def evaluate_model(self):
        """
        Evaluates the trained model on the test data and prints the accuracy.
        Note: This evaluates the last trained model.
        """
        print('Evaluating model...')
        loss, accuracy = self.model.evaluate(
            self.test_pad_sequences, 
            self.test_encoded_labels, 
            batch_size=self.hyperparams['batch_size']
        )
        print(f"Test accuracy: {accuracy}")
    
    def get_training_information(self):
        """
        Retrieves the training information containing average metrics and best model details.

        Returns:
            dict: A dictionary containing average training/validation metrics and best model's metrics.
        """
        return self.training_information