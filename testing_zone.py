import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

""" Logistic Regression Analysis"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

""" K_Nearest Neighbor Analysis"""
from sklearn.neighbors import KNeighborsClassifier

""" Functions from Other Files"""
from preprocess import fill_nan, preprocess_text
from gram_analysis import extract_top_bigrams, get_bigrams, calculate_similarity_score, evaluate_model, assign_grades_on_bell_curve, weighted_accuracy

import datetime as datetime


'''

    It is our goal to test the following: 
        1. A Logistic Regression Model
            a. Similarity score with top 1000 bigrams from base text
            b. Similarity score with top 1000 bigrams from noun, verb, and adjective text
            c. Dice similarity
            d. Jacquard Similarity

'''
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

    with open("results/results " + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".txt", 'a') as file:

        for prompt in prompts[pd.notna(prompts)]:
            file.write(f'\n==============\nAnalyzing prompt: {prompt}\n==============')

            train_df = df[df['prompt_name'] == prompt].copy()

            
            """ TOP 1000 BIGRAM SIM SCORES"""
            # Extract top 1000 bigrams from all essays
            top_bigrams = extract_top_bigrams(train_df['full_text'], n=n, NAV=False)

            # Calculate similarity scores for test data from top 1000 bigrams
            similarity_scores = []
            for essay in train_df['full_text']:
                tokens = preprocess_text(essay, NAV=False)
                essay_bigrams = get_bigrams(tokens)
                similarity = calculate_similarity_score(essay_bigrams, top_bigrams)
                similarity_scores.append(similarity)
            
            # converting to np array for them juicy easy functions. thank god for numpy
            similarity_scores = np.array(similarity_scores) 

            """ TOP 1000 NOUN-ADJ-VERB BIGRAM SIM SCORES"""

            # Extract top 1000 NAV bigrams from all essays
            top_bigrams = extract_top_bigrams(train_df['full_text'], n=n, NAV=False)

            nav_sim_scores = []
            for essay in train_df['full_text']:
                tokens = preprocess_text(essay, NAV=True)
                essay_bigrams = get_bigrams(tokens)
                similarity = calculate_similarity_score(essay_bigrams, top_bigrams)
                nav_sim_scores.append(similarity)

            nav_sim_scores = np.array(nav_sim_scores)

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

            # Calculate weights inversely proportional to class frequencies
            # class_weights = {
            #     label: total_samples / (len(class_counts) * count) 
            #     for label, count in class_counts.items()
            # }
            # Custom exponential weighting

            class_weights = {
                label: np.exp(-count/total_samples) 
                for label, count in class_counts.items()
            }

            # class_weights = {
            #     label: np.sqrt(total_samples/(count))
            #     for label, count in class_counts.items()
            # }
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
                
                model = LogisticRegression(class_weight=class_weights,     
                                        multi_class='multinomial', 
                                        solver=solver_lgreg,
                                        max_iter=3000,
                                        penalty="l2")
                                        
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

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

                # Evaluate accuracy
                accuracy = accuracy_score(y_test, y_pred, )
                file.write(f"Accuracy: {accuracy}")

                weighted_acc = weighted_accuracy(y_test, y_pred, weights=class_weights)
                print(f"Weighted Accuracy: {weighted_acc:.4f}")
                
            '''
            K-Nearest Neighbors
            '''
            if kNear:
                print(f"Conducting k-nearest neighbor analysis for prompt: {prompt}")
                file.write("\n\n--- K Nearest Neighbor Analysis ---\n")
                knn = KNeighborsClassifier(n_neighbors=6, metric=knnmetric)
                knn.fit(x_train, y_train)
                y_pred = knn.predict(x_test)

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