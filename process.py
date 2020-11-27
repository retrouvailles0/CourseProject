import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from urllib.request import urlopen
from bs4 import BeautifulSoup 
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import os
import sys
import time
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
import nltk


def stratified_k_fold():
    data = pd.read_csv("all_data.csv",sep=',')
    content_data = data['text'].values.astype('U')

    y = data['label'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    scores = []
    for train_indices, test_indices in skf.split(content_data, y):
        X_train = [content_data[li] for li in train_indices]
        Y_train = [y[li] for li in train_indices]
        X_test = [content_data[li] for li in test_indices]
        Y_test = [y[li] for li in test_indices]
   
        pipeline.fit(X_train, Y_train)
        score = pipeline.score(X_test, Y_test)
        scores.append(score)
        predictions = pipeline.predict(X_test)
        print(confusion_matrix(Y_test, predictions))
    score = sum(scores) / len(scores)
    print("StratifiedKFold Cross Validation Score - Using 5 folds")
    print(score)

df = pd.read_csv("all_data.csv", sep=',')
train,test = train_test_split(df,train_size=0.80)
train_data,test_data = pd.DataFrame(train),pd.DataFrame(test)
print("Successfully split the dataset into test and training dataset")
print(len(train_data.index))
print(len(test_data.index))
print("Train Data Report")
train_data_count = train_data.groupby('label')
print(train_data_count.count())
print("Test Data Report")
test_data_count = test_data.groupby('label')
print(test_data_count.count())
vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, ngram_range={1,2}, smooth_idf=True,
                             decode_error='ignore',  analyzer='word', lowercase=True, stop_words='english')
x_train_matrix = vectorizer.fit_transform(train_data['text'].values.astype('U'))
y_train_matrix = train_data['label'].values
y_test_matrix = test_data['label'].values
x_test_matrix = vectorizer.transform(test_data['text'].values.astype('U'))
pipeline = Pipeline([
  ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
  ('classifier', SGDClassifier())])
stratified_k_fold()