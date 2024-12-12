import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from textblob import TextBlob
from scipy.stats import ttest_ind, chi2_contingency
import pandas as pd

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Stereotypical/emotional keywords
stereotypical_keywords = ["sensitive", "stress", "anxious", "worry", "hormonal", "mood", "overwhelmed", "emotional", "therapy",
                          "relaxation", "yoga", "meditation", "therapy", "journaling", "mindfulness", "frail", "weak", "delicate", "dramatic",
                          "overreacting", "strong", "tough", "resilient", "manly", "stoic", "enduring"]

# Function: Count unique words
def count_unique_words(text):
    words = word_tokenize(text.lower())
    return len(set(words))

# Function: Count words and sentences
def count_words_sentences(text):
    words = len(word_tokenize(text))
    sentences = len(sent_tokenize(text))
    return words, sentences

# Function: Sentiment Analysis
def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)

# Function: Keyword Analysis
def count_keywords(text, keywords):
    words = word_tokenize(text.lower())
    keyword_counts = Counter(words)
    return sum(keyword_counts[word] for word in keywords if word in keyword_counts)

    # Function to calculate word lengths
def word_lengths(text):
    words = word_tokenize(text)
    lengths = [len(word) for word in words]
    if lengths:
        max_length = max(lengths)
        avg_length = sum(lengths) / len(lengths)
    else:
        max_length, avg_length = 0, 0
    return max_length, avg_length

# Function to calculate average sentence length (in words)
def avg_sentence_length(text):
    sentences = sent_tokenize(text)
    if sentences:
        words_per_sentence = [len(word_tokenize(sentence)) for sentence in sentences]
        avg_length = sum(words_per_sentence) / len(words_per_sentence)
    else:
        avg_length = 0
    return avg_length

# Analyze Responses
analysis_results = []
for response in responses:
    gender = response["gender"]
    text = response["text"]

    # Existing metrics
    unique_words = count_unique_words(text)
    words, sentences = count_words_sentences(text)
    sentiment = sentiment_analysis(text)
    keyword_count = count_keywords(text, stereotypical_keywords)

    # New metrics
    max_word_length, avg_word_length = word_lengths(text)
    avg_sentence_len = avg_sentence_length(text)

    # Append results
    analysis_results.append({
        "Gender": gender,
        "Unique Words": unique_words,
        "Word Count": words,
        "Sentence Count": sentences,
        "Sentiment": sentiment,
        "Keyword Count": keyword_count,
        "Max Word Length": max_word_length,
        "Avg Word Length": avg_word_length,
        "Avg Sentence Length": avg_sentence_len
    })

# Convert to DataFrame for easy analysis
df = pd.DataFrame(analysis_results)

# Separate data by gender
male_data = df[df["Gender"] == "male"]
female_data = df[df["Gender"] == "female"]

male_data = male_data.dropna()
female_data = female_data.dropna()

from scipy.stats import ttest_ind, mannwhitneyu

# Function to perform T-test with error handling
def perform_ttest(group1, group2, label):
    if len(group1) > 1 and len(group2) > 1:
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        return t_stat, p_val
    else:
        print(f"Insufficient data for T-test on {label}")
        return None, None

# Perform T-test on Word Count
t_stat_word, p_val_word = perform_ttest(male_data["Word Count"], female_data["Word Count"], "Word Count")

# Perform T-test on Sentiment
t_stat_sentiment, p_val_sentiment = perform_ttest(male_data["Sentiment"], female_data["Sentiment"], "Sentiment")

# Chi-square (already safe for small datasets)
contingency_table = pd.crosstab(df["Gender"], df["Keyword Count"])
chi2, p_val_chi2, _, _ = chi2_contingency(contingency_table)

# Output results
print("\nStatistical Tests:")
print(f"T-test (Word Count): t-stat = {t_stat_word}, p-value = {p_val_word}")
print(f"T-test (Sentiment): t-stat = {t_stat_sentiment}, p-value = {p_val_sentiment}")
print(f"Chi-square (Keyword Count): chi2 = {chi2}, p-value = {p_val_chi2}")

# Print results
print("Analysis Results")
print(df)

print("\nStatistical Tests:")
print(f"T-test (Word Count): t-stat = {t_stat_word}, p-value = {p_val_word}")
print(f"T-test (Sentiment): t-stat = {t_stat_sentiment}, p-value = {p_val_sentiment}")
print(f"Chi-square (Keyword Count): chi2 = {chi2}, p-value = {p_val_chi2}")