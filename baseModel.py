
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
from sklearn.preprocessing import LabelEncoder
import random
import statistics


""" Logistic Regression Analysis"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


""" K_Nearest Neighbor Analysis"""
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import confusion_matrix

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate the quadratic weighted kappa between two sets of ratings.
    
    Parameters:
    y_true (array-like): Array of true ratings (ground truth)
    y_pred (array-like): Array of predicted ratings
    
    Returns:
    float: The quadratic weighted kappa score
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # Verify inputs are valid (between 1 and 6)
    if not (np.all(np.isin(y_true, range(1, 7))) and np.all(np.isin(y_pred, range(1, 7)))):
        raise ValueError("All values must be between 1 and 6")
    
    # Calculate confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=range(1, 7))
    
    # Calculate weights matrix
    num_classes = 6
    weights = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i + 1) - (j + 1)) ** 2
    
    # Calculate expected matrix
    row_sum = np.sum(conf_mat, axis=1)
    col_sum = np.sum(conf_mat, axis=0)
    expected = np.outer(row_sum, col_sum) / np.sum(row_sum)
    
    # Calculate weighted matrices
    weighted_matrix = weights * conf_mat
    weighted_expected = weights * expected
    
    # Calculate kappa
    observed = np.sum(weighted_matrix)
    expected_weighted = np.sum(weighted_expected)
    total = np.sum(conf_mat)
    
    # Handle edge case where all predictions are the same
    if expected_weighted == 0:
        return 1.0 if observed == 0 else 0.0
    
    kappa = 1 - (observed / expected_weighted)
    
    return kappa

# Example usage:
if __name__ == "__main__":
    # Example data
    true_scores = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    pred_scores = [1, 2, 3, 3, 5, 6, 2, 2, 2, 4]
    
    score = quadratic_weighted_kappa(true_scores, pred_scores)
    print(f"Quadratic Weighted Kappa: {score:.4f}")

def evaluate_model(predicted_grades, actual_grades):
    accuracy = accuracy_score(actual_grades, predicted_grades)
    precision = precision_score(actual_grades, predicted_grades, average='weighted')
    recall = recall_score(actual_grades, predicted_grades, average='weighted')
    f1 = f1_score(actual_grades, predicted_grades, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def weighted_accuracy(y_true, y_pred, weights=None):

    # If no weights provided, use uniform weights
    if weights is None:
        return accuracy_score(y_true, y_pred)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create weight array matching the true labels
    sample_weights = np.array([weights[label] for label in y_true])
    
    # Calculate weighted accuracy
    correct = (y_true == y_pred)
    return np.sum(correct * sample_weights) / np.sum(sample_weights)

    

def baseModel(df, prompts):
    for prompt in prompts[pd.notna(prompts)]:
        print(f"\n\n\n Working on prompt {prompt}")
        df_subset = df[df['prompt_name'] == prompt]

        # Then create X and y from the subset
        X = df_subset.drop(['essay_id', 'score', 'full_text'], axis=1)
        y = df_subset['score']
        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

        class_counts = y_train.value_counts()
        total_samples = len(y_train)

       
        # Custom exponential weighting

        class_weights = {
            label: np.exp(-count/total_samples) 
            for label, count in class_counts.items()
        }

        model = LogisticRegression(class_weight=class_weights,     
                                        multi_class='multinomial', 
                                        solver="newton-cg",
                                        max_iter=3000,
                                        penalty=None,
                                        random_state=7362)
                                        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        qwk_score = quadratic_weighted_kappa(y_test, y_pred)
        print(f"Quadratic Weighted Kappa Score: {qwk_score:.4f}")

        weighted_acc = weighted_accuracy(y_test, y_pred, weights=class_weights)
        print(f"Weighted Accuracy: {weighted_acc:.4f}")


        print(evaluate_model(y_pred, y_test))
