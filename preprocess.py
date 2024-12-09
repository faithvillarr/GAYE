from nltk.tokenize import word_tokenize
import nltk
import re
import random
import statistics
import language_tool_python
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix




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
def preprocess_text(text, NAV = False):
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    # Only process words that are nouns, verbs and adjectives
    if NAV:
        tagged = nltk.pos_tag(tokens)
        tags_to_keep = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NP', 'NPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        tokens = [word for word, tag in tagged if tag in tags_to_keep]

    return tokens

def main():
    temp = preprocess_text("The Dice coefficient has several advantages over other similarity metrics. It is particularly useful for imbalanced datasets, where one set may be much larger than the other. It is a better choice for image segmentation tasks, as it is more sensitive to overlap between the predicted and ground truth masks. This is achieved by treating the segmentation masks as sets of pixels. The predicted segmentation and the ground truth segmentation are both represented as binary masks, where a pixel is either part of the segmented object or not.", 
                    NAV = True)
    print(temp)

if __name__ == "__main__":
    main()