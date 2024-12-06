import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
import statistics
import language_tool_python


np.random.seed(2024)
import nltk
nltk.download('punkt_tab')
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

def main():
    df = pd.read_csv("ASAP2_competitiondf_with-metadata_TheLearningExchange-trainonly.csv")

    prompts = df['prompt_name'].unique()

    '''
    Hyper parameters to experiment with:
    '''
    alambda = 1 # Strengthen or weakens std dev when calculating z-scores.
    n = 500 # Number of top bi grams to consider

    for prompt in prompts:
        
        print(f'\nAnalyzing prompt: {prompt}')

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
        
        similarity_scores = np.array(similarity_scores)
        df.loc[df['prompt_name'] == prompt, 'similarity_score'] = similarity_scores
        
        
        # Assign and evaluate grades
        predicted_grades = assign_grades_on_bell_curve(similarity_scores, alambda)
        actual_grades = train_df['score'].values
        
        # Calculate metrics
        metrics = evaluate_model(predicted_grades, actual_grades)
        
        # Print results
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Print grade distribution
        print("\nGrade Distribution:")
        grade_dist = print(pd.DataFrame({
            'Predicted': pd.Series(predicted_grades).value_counts().sort_index(),
            'Actual': pd.Series(actual_grades).value_counts().sort_index()
        }))

        write_results_to_file(prompt, metrics, grade_dist)

    # df = addFeatures(df)
    # csv_path = "FeaturesAdded.csv"
    # df.to_csv(csv_path, index=False)
    # print(f"DataFrame saved to CSV: {csv_path}")


if __name__ == "__main__":
    main()