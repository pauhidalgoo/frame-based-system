import tensorflow.keras.backend as K


def macro_f1(y_true, y_pred):
    """
    Calculate macro F1 score for multi-class classification matching sklearn's implementation.
    
    Args:
        y_true: One-hot encoded true labels (batch_size, ..., num_classes)
        y_pred: Predicted probability distributions (batch_size, ..., num_classes)
    """
    # Get predicted classes (one-hot encoding of the predicted classes)
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), K.int_shape(y_pred)[-1])
    
    # Flatten tensors for per-class calculation
    y_true_flat = K.reshape(y_true, (-1, K.int_shape(y_true)[-1]))
    y_pred_flat = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    
    # Calculate per-class metrics: true positives, false positives, and false negatives
    true_positives = K.sum(y_true_flat * y_pred_flat, axis=0)
    false_positives = K.sum(y_pred_flat * (1 - y_true_flat), axis=0)
    false_negatives = K.sum(y_true_flat * (1 - y_pred_flat), axis=0)
    
    # Calculate precision and recall per class
    precision = true_positives / (true_positives + false_positives + K.epsilon())
    recall = true_positives / (true_positives + false_negatives + K.epsilon())
    
    # Calculate F1 score per class
    f1_scores = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
    # Handle cases where precision + recall = 0
    zero_mask = K.equal(precision + recall, 0)
    f1_scores = K.switch(zero_mask, K.zeros_like(f1_scores), f1_scores)
    
    # Calculate the macro F1 score: mean of per-class F1 scores
    return K.mean(f1_scores)

# Test function that uses sklearn directly
def test_macro_f1(y_true, y_pred):
    """
    Test function using sklearn's f1_score
    """
    import numpy as np
    from sklearn.metrics import f1_score
    
    y_true_classes = np.argmax(y_true, axis=-1)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    
    return f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)


import numpy as np
from sklearn.metrics import f1_score, classification_report

# Example with class imbalance
y_true = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

y_pred = np.array([
    [0.9, 0.1, 0.0],
    [0.8, 0.1, 0.1],
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])



# Test with sklearn directly
y_true_classes = np.argmax(y_true, axis=-1)
y_pred_classes = np.argmax(y_pred, axis=-1)
sklearn_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
print("Sklearn F1:", sklearn_f1)
own_f1 = macro_f1(y_true, y_pred)
print("own F1:", sklearn_f1)
print("\nFull classification report:")
print(classification_report(y_true_classes, y_pred_classes))