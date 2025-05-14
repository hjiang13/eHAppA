
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

# Load data
keywords_df = pd.read_csv('keyFeature.csv')
labels_df = pd.read_csv('PARIS_result.csv')
keywords_df = keywords_df.sort_values("Filename").reset_index(drop=True)
labels_df = labels_df.sort_values("BenchMark").reset_index(drop=True)

# Preprocessing
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
X = vectorizer.fit_transform(keywords_df['Keywords']).toarray()
vocab = vectorizer.get_feature_names_out()

# Count all tokens

token_counts = np.asarray(X.sum(axis=0)).flatten()
token_names = vectorizer.get_feature_names_out()

# Identify and replace tokens that occur only once
tokens_to_replace = [token_names[i] for i in range(len(token_names)) if token_counts[i] == 1]

def replace_single_occurrence_tokens(text):
    tokens = text.split(',')
    replaced_tokens = [token if token not in tokens_to_replace else 'NaN' for token in tokens]
    return ','.join(replaced_tokens)

keywords_df["Filtered Keywords"] = keywords_df["Keywords"].apply(replace_single_occurrence_tokens)

# Display the result
print(keywords_df)

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
X = vectorizer.fit_transform(keywords_df["Filtered Keywords"]).toarray()


vocab = vectorizer.get_feature_names_out()
# Compute point-biserial correlation
correlations = [pointbiserialr(X[:, i], labels_df['Actual_SDC'])[0] for i in range(X.shape[1])]

# Create DataFrame
df = pd.DataFrame({'Token': vocab, 'Correlation': correlations})
df['Importance'] = abs(df['Correlation'])
df = df.sort_values('Importance', ascending=False)
print(df)

df.to_csv('featureImportance_SDC.csv', index=False)