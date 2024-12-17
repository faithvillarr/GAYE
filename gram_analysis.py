import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
from sklearn.preprocessing import LabelEncoder
import random
import statistics
from baseModel import baseModel
from sklearn.metrics import confusion_matrix



""" Logistic Regression Analysis"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


""" K_Nearest Neighbor Analysis"""
from sklearn.neighbors import KNeighborsClassifier


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


# Given tokens from preproces_text(), return a list of bigram tuples. 
def get_bigrams(tokens):
    return [(tokens[i], tokens[i+1]) for i in range(0, len(tokens)-1)]

# Given tokens from preproces_text(), return a list of trigram tuples. 
def get_trigrams(tokens):
    return [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(0, len(tokens)-2)]

# Extracts the top n most common bigrams from all essays.
# Returns a list of bigram tuples.
def extract_top_bigrams(essays, n=1000, NAV = False):
    all_bigrams = []
    for essay in essays:
        tokens = preprocess_text(essay, NAV)
        bigrams = get_bigrams(tokens)
        all_bigrams.extend(bigrams)
    
    # Count and get top n bigrams
    bigram_counts = Counter(all_bigrams)
    return [bigram for bigram, count in bigram_counts.most_common(n)]

# Calculates similarity score between essay and top bigrams encountered. 
def calculate_similarity_score(essay_bigrams, top_bigrams):
    common_bigrams = set(essay_bigrams).intersection(set(top_bigrams))
    similarity = len(common_bigrams) / len(top_bigrams) if top_bigrams else 0
    return similarity


""" Bell Curve Analysis """
# Assigns grades (1-5) based on similarity scores (np.array) using a bell curve.
def assign_grades_on_bell_curve(similarity_scores: np.array, alambda = 1):
    
    # Calculate mean, standard deviation and z-score for bell curve. 
    mean = np.mean(similarity_scores)
    std = np.mean(similarity_scores)
    z_scores = (similarity_scores - mean) / (std * alambda)
    
    # Assign letter grades based on z-scores
    grades = np.empty(len(z_scores), dtype=int)
    grades[z_scores >= 2.0] = 6
    grades[(z_scores >= 1.5) & (z_scores < 2.0)] = 5
    grades[(z_scores >= 0.5) & (z_scores < 1.5)] = 4
    grades[(z_scores >= -0.5) & (z_scores < 0.5)] = 3
    grades[(z_scores >= -1.5) & (z_scores < -0.5)] = 2
    grades[z_scores < -1.5] = 1
    
    return grades

# Calculate evaluation metrics by comparing predicted vs actual grades.
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

def write_results_to_file(prompt_name, metrics, grade_dist_df):
    # need to write 
    return 

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

from datetime import datetime
from preprocess import preprocess_text, fill_nan

def main():
    dfBASE = pd.read_csv("ASAP2_competitiondf_with-metadata_TheLearningExchange-trainonly.csv")
    df = pd.read_csv("FeaturesAdded.csv")
    print(f"There are {df.isna().any(axis=1).sum()} rows with nan")
    df = fill_nan(df)
    dfBASE = fill_nan(dfBASE)

    le = LabelEncoder() #encodes the target variable

    toEncode = ['assignment', 'prompt_name', "economically_disadvantaged", 'student_disability_status', 'ell_status', 'race_ethnicity', 'gender'] 

    for item in toEncode:
        # print(item)
        # print(df)
        df[item] = le.fit_transform(df[item])
        dfBASE[item] = le.fit_transform(dfBASE[item])
        # for i, label in enumerate(le.classes_):
        #     print(f"{i} -> {label}")

    prompts = df['prompt_name'].unique()
    prompts.sort()
    baseModel(dfBASE, prompts)

    '''
    Select Analysis Type
    '''
    bell_curve = False
    log_reg = True
    kNear = False
    embedding = False
    neural_net = False

    '''
    Hyper parameters to experiment with:
    '''
    alambda = 1 # Strengthen or weakens std dev when calculating z-scores.
    n = 1000 # Number of top bi grams to consider
    
    test_size = 0.2 # % of data set to be used to train per prompt
    solver_lgreg = 'newton-cg'

    knnmetric = 'euclidean'

    with open("results/results " + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".txt", 'a') as file:

        for prompt in prompts[pd.notna(prompts)]:
            file.write(f'\n==============\nAnalyzing prompt: {prompt}\n==============')

            train_df = df[df['prompt_name'] == prompt].copy()


            # Extract top 1000 bigrams from all essays
            top_bigrams = extract_top_bigrams(train_df['full_text'], n=n)
            
            # Calculate similarity scores for test data
            similarity_scores = []
            for essay in train_df['full_text']:
                tokens = preprocess_text(essay)
                essay_bigrams = get_bigrams(tokens)
                similarity = calculate_similarity_score(essay_bigrams, top_bigrams)
                similarity_scores.append(similarity)
            
            # converting to np array for them juicy easy functions. thank go for numpy
            similarity_scores = np.array(similarity_scores) 

            # Drop variables that score not be found from full_text
            # columns_to_drop = [
            #     "essay_id",
            #     "full_text",
            #     "assignment",
            #     "prompt_name",
            #     # "economically_disadvantaged",
            #     # "student_disability_status",
            #     # "ell_status",
            #     # "race_ethnicity",
            #     # "gender",
            #     # "grade_level"
            # ]
            # train_df = train_df.drop(columns=columns_to_drop)
            
            # # Stack sim scores and word count
            # prompt_arr = np.column_stack((similarity_scores, np.array(train_df['essay_word_count'])))
            

            df_subset = df[df['prompt_name'] == prompt]

            # Then create X and y from the subset
            X = df_subset.drop(['essay_id', 'score', 'full_text'], axis=1)
            y = df_subset['score']
            x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42, stratify=y)


            

            


            class_counts = y_train.value_counts()
            total_samples = len(y_train)

        
            # Custom exponential weighting

            class_weights = {
                label: np.exp(-count/total_samples) 
                for label, count in class_counts.items()
            }

          
            '''
            Bell Curve Analysis and performance
            '''
            if bell_curve: 
                print(f"Conducting bell curve analysis for prompt: {prompt}")
                file.write("\n\n--- Bell Curve Analysis ---\n")
                predicted_grades = assign_grades_on_bell_curve(similarity_scores, alambda)
                actual_grades = train_df['score'].values

                metrics = evaluate_model(predicted_grades, actual_grades)
                file.write("Evaluation Metrics:\n")
                for metric, value in metrics.items():
                    file.write(f"{metric}: {value:.4f}\n")
                file.write("\nGrade Distribution:")
                file.write(pd.DataFrame({
                    'Predicted': pd.Series(predicted_grades).value_counts().sort_index(),
                    'Actual': pd.Series(actual_grades).value_counts().sort_index()
                }).to_string())
            '''
            Logistic Regression Model
            '''
            if log_reg:
                print(f"Conducting logistic Regression for prompt: {prompt}")
                file.write("\n--- Logistic Regression Analysis ---\n")
                # Stack sim scores and word count
                model = LogisticRegression(class_weight=class_weights,     
                                        multi_class='multinomial', 
                                        solver='newton-cg',
                                        max_iter=3000,
                                        penalty=None,
                                        random_state=7362)
                                        
           
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)


                qwk_score = quadratic_weighted_kappa(y_test, y_pred)
                print(f"Quadratic Weighted Kappa Score: {qwk_score:.4f}")


                weighted_acc = weighted_accuracy(y_test, y_pred, weights=class_weights)
                print(f"Weighted Accuracy: {weighted_acc:.4f}")


                metrics = evaluate_model(y_pred, y_test)
                file.write("Evaluation Metrics:\n")
                for metric, value in metrics.items():
                    file.write(f"{metric}: {value:.4f}\n")

                file.write("\nGrade Distribution:")
                file.write(pd.DataFrame({
                    'Predicted': pd.Series(y_pred).value_counts().sort_index(),
                    'Actual': pd.Series(y_test).value_counts().sort_index()
                }).to_string())

                # probabilities = model.predict_proba(x_test)
                # file.write(f"Class Probabilities:\n{probabilities[:5]}")

                # Evaluate accuracy
                accuracy = accuracy_score(y_test, y_pred, )
                file.write(f"Accuracy: {accuracy}")

                weighted_acc = weighted_accuracy(y_test, y_pred, weights=class_weights)
                print(f"Weighted Accuracy: {weighted_acc:.4f}")
                # # Evaluate accuracy
                # accuracy = accuracy_score(y_test, y_pred)
                # file.write(f"Accuracy: {accuracy}")

                # # Classification report
                # file.write("Classification Report:")
                # file.write(classification_report(y_test, y_pred))
            '''
            K-Nearest Neighbors
            '''
            if kNear:
                print(f"Conducting k-nearest neighbor analysis for prompt: {prompt}")
                file.write("\n\n--- K Nearest Neighbor Analysis ---\n")
                knn = KNeighborsClassifier(n_neighbors=6, metric=knnmetric)
                knn.fit(x_train, y_train)
                y_pred = knn.predict(x_test)
                # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
                # file.write(f"\nClassification Report:\n {classification_report(y_test, y_pred)}")

                metrics = evaluate_model(y_pred, y_test)
                file.write("Evaluation Metrics:\n")
                for metric, value in metrics.items():
                    file.write(f"{metric}: {value:.4f}\n")

                file.write("\nGrade Distribution:")
                file.write(pd.DataFrame({
                    'Predicted': pd.Series(y_pred).value_counts().sort_index(),
                    'Actual': pd.Series(y_test).value_counts().sort_index()
                }).to_string())



if __name__ == "__main__":
    main()