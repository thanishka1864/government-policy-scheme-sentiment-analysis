# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:30:25 2026

@author: thani_cohvxow
"""

import pandas as pd
import re
import html
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = r"C:\Users\thani_cohvxow\Downloads\demonetization-tweets.csv"

df = pd.read_csv(CSV_PATH, encoding="cp1252")

df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

def clean_tweet(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_tweet)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["clean_text"].apply(get_sentiment)

print("\nSENTIMENT DISTRIBUTION")
print(df["sentiment"].value_counts())

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("svm", LinearSVC(random_state=42))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nSVM MODEL ACCURACY")
print(round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Negative", "Neutral", "Positive"]
))

df["predicted_sentiment"] = model.predict(df["clean_text"])

output_path = r"C:\Users\thani_cohvxow\Downloads\demonetization_sentiment_results.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nResults saved successfully to:")
print(output_path)

print("\nSAMPLE PREDICTIONS")
print(
    df[["text", "sentiment", "predicted_sentiment"]]
    .head(10)
    .to_string(index=False)
)

while True:
    tweet = input("\nEnter a tweet to analyze, or type exit: ")

    if tweet.strip().lower() == "exit":
        break

    cleaned_tweet = clean_tweet(tweet)
    result = model.predict([cleaned_tweet])[0]

    print("Predicted Sentiment:", result)