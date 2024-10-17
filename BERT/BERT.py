import pandas as pd
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification, create_optimizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


def format_labels(labels):
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

# Load the data
train_data = pd.read_csv('./baiges/data/train.csv', header=None)
test_data = pd.read_csv('./baiges/data/test.csv', header=None)
val_data = train_data.tail(900)
train_data = pd.read_csv('./baiges/data/train.csv', header=None, nrows=4078)

# Sentences
train_sentences = train_data[0].tolist()
val_sentences = val_data[0].tolist()
test_sentences = test_data[0].tolist()

# To string
train_sentences = [str(sentence) for sentence in train_sentences]  # Convert all elements to string if needed
val_sentences = [str(sentence) for sentence in val_sentences]  # Convert all elements to string if needed
test_sentences = [str(sentence) for sentence in test_sentences]  # Convert all elements to string if needed

# Labels
train_labels = format_labels(train_data[2].tolist())
val_labels = format_labels(val_data[2].tolist())
test_labels = format_labels(test_data[2].tolist())

values_to_remove = ['day_name','airfare+flight','flight+airline','flight_no+airline']

def remove_values_and_indices(values_to_remove, labels_list, sentence_list):
        """
        Removes specified values and their corresponding indices from labels and sequences.

        Args:
            values_to_remove (list): A list of label values to remove.
            labels_list (list): The list of labels.
            sentence_list (list): The list of the sequences.

        Returns:
            tuple: A tuple containing the cleaned labels list and the cleaned padded sequences array.
        """
        indices_to_remove = [idx for idx, item in enumerate(labels_list) if item in values_to_remove]
        cleaned_labels_list = [item for item in labels_list if item not in values_to_remove]
        cleaned_sentence_list = [item for idx, item in enumerate(sentence_list) if idx not in indices_to_remove]
        return cleaned_labels_list, cleaned_sentence_list

val_labels, val_sentences = remove_values_and_indices(values_to_remove, val_labels, val_sentences)
test_labels, test_sentences = remove_values_and_indices(values_to_remove, test_labels, test_sentences)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)  # Fit on training labels

test_sentences = list(test_sentences)

# Transform labels
train_labels_encoded = label_encoder.transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)


# Ensure labels are numpy arrays of integer type
train_labels_encoded = np.array(train_labels_encoded, dtype='int32')
val_labels_encoded = np.array(val_labels_encoded, dtype='int32')
test_labels_encoded = np.array(test_labels_encoded, dtype='int32')

# Update num_classes
num_classes = len(label_encoder.classes_)

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(
    train_sentences, truncation=True, padding=True, max_length=128
)

val_encodings = tokenizer(
    val_sentences, truncation=True, padding=True, max_length=128
)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels_encoded
)).shuffle(len(train_sentences)).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels_encoded
)).batch(32)

# Load the pre-trained model
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=num_classes
)

# Calculate steps per epoch
epochs = 3  # You can adjust this
steps_per_epoch = len(train_dataset)
num_train_steps = steps_per_epoch * epochs

# Create the optimizer
optimizer, lr_schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01
)

# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Save the model in HDF5 format
model.save_pretrained('MODEL.h5')  # Specify the filename with .h5 extension

# Evaluate the model on the test set
test_encodings = tokenizer(
    test_sentences, truncation=True, padding=True, max_length=128
)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels_encoded
)).batch(32)


loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.2f}")