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
import language_tool_python


""" Logistic Regression Analysis"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


""" K_Nearest Neighbor Analysis"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
                'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
                'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
                'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
                'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
def addFeatures(df):
    
    connectives =  [
    "after",
    "earlier", 
    "before",
    "during",
    "while",
    "later",
    "because",
    "consequently",
    "thus",
    "both",
    "additionally",
    "furthermore",
    "moreover",
    "actually",
    "as a result",
    "due to",
    "but",
    "yet",
    "however",
    "although",
    "nevertheless"
]
    tool = language_tool_python.LanguageTool('en-US')
    def get_features(text, index=None, total=None):
        if isinstance(index, tuple):
            index = index[0]
        if index is not None and total is not None:
            if index % 100 == 0:
                print(f"Processing essay {index}/{total} ({(index/total*100):.1f}%)")
        text = str(text)  
        sentences = re.split(r'[.!?]+(?=\s+|$)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = statistics.mean([len(word) for word in words]) if words else 0
        num_commas = text.count(',')
        num_periods = text.count('.')
        num_semicolons = text.count(';')
        num_exclamations = text.count('!')
        num_questions = text.count('?')
        
        text_lower = text.lower()
        num_connectives = sum(text_lower.count(conn.lower()) for conn in connectives)
        matches = tool.check(text)
        spelling_errors = sum(1 for match in matches if match.ruleId.startswith('MORFOLOGIK_'))
        grammar_errors = len(matches) - spelling_errors
        return {
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'num_commas': num_commas,
            'num_periods': num_periods,
            'num_semicolons': num_semicolons,
            'num_exclamations': num_exclamations,
            'num_questions': num_questions,
            'num_connectives': num_connectives,
            'num_spelling_errors': spelling_errors,
            'num_grammar_errors': grammar_errors
        }
    total_essays = len( df)
    features = df['full_text'].reset_index().apply(
    lambda row: get_features(row['full_text'], row.name, total_essays), 
    axis=1
)
    for feature_name in features.iloc[0].keys():
        df[feature_name] = features.apply(lambda x: x[feature_name])
    return df


def fill_nan(df):
    df.dropna(subset=['full_text'], inplace=True) #drops rows without "full_text" 

    for idx in df.index: #fill in empty essay_word_count
        if pd.isna(df.loc[idx, 'essay_word_count']):
            df.loc[idx, 'essay_word_count'] = len(str(df.loc[idx, 'full_text']).split())



    def sameAsAboveBelow(feature, idx): #manually checked that idx = 0 and idx = len(df) are complete rows, so don't need to consider this case
        return df[feature].iloc[idx-1] == df[feature].iloc[idx+1]

    mask = df['assignment'].notna()
    df.loc[mask] = df.loc[mask].sort_values(by='assignment') #sorts by "assignment", but leaves rows with a blank assignment where they are. VERIFIED 

    missing_indices = df[df["assignment"].isna()].index
    for idx in missing_indices.sort_values():
        if sameAsAboveBelow("assignment", idx):
            df.loc[idx, "assignment"] = df["assignment"].iloc[idx-1] #function returned True, take the value from the row above
        else:
            if pd.isna(df.loc[idx, "prompt_name"]): #prompt_name doesn't exist either, default to row above
                df.loc[idx, "assignment"] = df["assignment"].iloc[idx-1]
                df.loc[idx, "prompt_name"] = df["prompt_name"].iloc[idx-1]

            else: #prompt_name exists
                if df.loc[idx, "prompt_name"] == df["prompt_name"].iloc[idx+1]: #matches row below, copy its data
                    df.loc[idx, "assignment"] = df["assignment"].iloc[idx+1]
                else:
                    df.loc[idx, "assignment"] = df["assignment"].iloc[idx-1] #matches row above (or row below doesn't exist), use row above



    mask = df['prompt_name'].notna()
    df.loc[mask] = df.loc[mask].sort_values(by='prompt_name') #sorts by "assignment", but leaves rows with a blank assignment where they are.


    missing_indices = df[df["prompt_name"].isna()].index #do the same first check as we did for "assignment"
    for idx in missing_indices.sort_values():
        if sameAsAboveBelow("prompt_name", idx):
            df.loc[idx, "prompt_name"] = df["prompt_name"].iloc[idx-1]


    mask = df['grade_level'].notna()
    df.loc[mask] = df.loc[mask].sort_values(by='grade_level')# must sort by grade first for the rule to work properly

    missing_indices = df[df["grade_level"].isna()].index
    for idx in missing_indices.sort_values():
        df.loc[idx, "grade_level"] = df["grade_level"].iloc[idx-1] #default to the row above
                    


    featuresToCheck = [ 
        ("economically_disadvantaged","Economically disadvantaged", "Economically disadvantaged"),
        ("student_disability_status","Identified as having disability", "Not identified as having disability"),
        ("ell_status", "Yes", "No"),
        ("gender", 'M', 'F')
        ] #feautre, option1, option2 

    #I could just encode these categories right now to save time and lines of code, but I worry we may want to see the whole dataset again before any encoding
    for feature, option1, option2 in featuresToCheck:
        missing_indices = df[df[feature].isna()].index
        for idx in missing_indices.sort_values():
            if random.choice([True, False]):
                df.loc[idx, feature] = option1
            else:
                df.loc[idx, feature] = option2


    uniqueRaceEthnicity = [x for x in df['race_ethnicity'].unique() if not pd.isna(x)] 

    missing_indices = df[df["race_ethnicity"].isna()].index
    for idx in missing_indices.sort_values():
        df.loc[idx, "race_ethnicity"] = random.choice(uniqueRaceEthnicity) #fill in the cell with a random race_ethnicity


    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"There are now {df.isna().any(axis=1).sum()} rows with nan")
    return df


def calculate_qwk(y_true, y_pred, n_classes):
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.asarray(y_true) - 1  # Shift class labels to start from 0
    y_pred = np.asarray(y_pred) - 1  # Shift class labels to start from 0
    
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")
    if len(y_true) == 0:
        raise ValueError("Arrays cannot be empty")
    
    # Create confusion matrix with adjusted labels
    conf_mat = confusion_matrix(y_true, y_pred, 
                              labels=list(range(n_classes-1)))  # Adjust range
    
    # Create weight matrix
    weights = np.zeros((n_classes-1, n_classes-1))
    for i in range(n_classes-1):
        for j in range(n_classes-1):
            weights[i,j] = ((i+1)-(j+1)) ** 2  # Adjust weight calculation
    
    # Calculate row and column sums
    row_sums = conf_mat.sum(axis=1)
    col_sums = conf_mat.sum(axis=0)
    total = np.sum(row_sums)
    
    # Handle edge case where total is 0
    if total == 0:
        return 0.0
    
    # Calculate expected matrix (with safe division)
    expected = np.outer(row_sums, col_sums) / total
    
    # Calculate weighted matrices
    w_observed = np.sum(weights * conf_mat)
    w_expected = np.sum(weights * expected)
    
    # Calculate denominator safely
    denominator = np.sum(weights * np.outer(row_sums, row_sums)) / total - w_expected
    
    # Handle edge case where denominator is 0
    if abs(denominator) < 1e-10:  # Using small threshold instead of exact 0
        return 1.0 if w_observed == w_expected else 0.0
    
    # Calculate QWK
    return 1 - ((w_observed - w_expected) / denominator)



# Preprocesses text.
def preprocess_text(text):
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

# Given tokens from preproces_text(), return a list of bigram tuples. 
def get_bigrams(tokens):
    return [(tokens[i], tokens[i+1]) for i in range(0, len(tokens)-1)]

# Given tokens from preproces_text(), return a list of trigram tuples. 
def get_trigrams(tokens):
    return [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(0, len(tokens)-2)]

# Extracts the top n most common bigrams from all essays.
# Returns a list of bigram tuples.
def extract_top_bigrams(essays, n=1000):
    all_bigrams = []
    for essay in essays:
        tokens = preprocess_text(essay)
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
def main():
    # df = pd.read_csv("ASAP2_competitiondf_with-metadata_TheLearningExchange-trainonly.csv")
    df = pd.read_csv("FeaturesAdded.csv")
    print(f"There are {df.isna().any(axis=1).sum()} rows with nan")
    df = fill_nan(df)

    le = LabelEncoder() #encodes the target variable

    toEncode = ['assignment', 'prompt_name', "economically_disadvantaged", 'student_disability_status', 'ell_status', 'race_ethnicity', 'gender'] 

    for item in toEncode:
        # print(item)
        # print(df)
        df[item] = le.fit_transform(df[item])

    # print(f"\n{item} mapping:") #prints out what category converted to what integer
    # for i, label in enumerate(le.classes_):
    #     print(f"{i} -> {label}")


    prompts = df['prompt_name'].unique()
    prompts.sort()
    # print(prompts)

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
    solver_lgreg = 'saga'

    knnmetric = 'euclidean'

    with open("results " + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".txt", 'a') as file:

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
            x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)

            class_counts = y_train.value_counts()
            total_samples = len(y_train)

            # Calculate weights inversely proportional to class frequencies
            class_weights = {
                label: total_samples / (len(class_counts) * count) 
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
                prompt_arr = np.column_stack((similarity_scores, np.array(train_df['essay_word_count'])))
                # X = df.drop('score', axis=1)
                # y = df['score']
                # x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)
                #  Due to the bell-curved nature of our dataset, we use balanced 
                #   weight classes to make prediction of minority classes more likely. 
                # model = LogisticRegression(class_weight=class_weights,     
                # prompt_arr = np.column_stack((similarity_scores, np.array(train_df['essay_word_count'])))
                # prompt_arr = train_df
                # x_train, x_test, y_train, y_test = train_test_split(prompt_arr, train_df['score'], test_size=test_size, random_state=42)
                class_counts = y_train.value_counts()
                total_samples = len(y_train)

                # Calculate weights inversely proportional to class frequencies
                class_weights = {
                    label: total_samples / (len(class_counts) * count) 
                    for label, count in class_counts.items()
}
                model = LogisticRegression(class_weight=class_weights,     
                                        multi_class='multinomial', 
                                        solver=solver_lgreg,
                                        max_iter=3000,
                                        penalty="l1")
                                        
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)


                # qwk_score = calculate_qwk(y_test, y_pred, n_classes=7)
                # print(f"Quadratic Weighted Kappa Score: {qwk_score:.4f}")


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