import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import  MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier, LogisticRegression
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
import random

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

X_train = train_data['text'].values.astype('U')
Y_train = train_data['label'].values
X_test = test_data['text'].values.astype('U')
Y_test = test_data['label'].values
models = [SVC(), LinearSVC(), MultinomialNB(), DecisionTreeClassifier(), SGDClassifier(), AdaBoostClassifier()]

max_score = 0
pipeline = None
def create_pipeline(model):
    global pipeline
    global max_score
    print(model)
    pp = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', model)])
    pp.fit(X_train, Y_train)
    score = pp.score(X_test, Y_test)
    print(score)
    if (score > max_score):
        max_score = score
        pipeline = pp

for model in models:
    create_pipeline(model)
print("Model with highest score: ")
print(pipeline)

joblib.dump(pipeline, 'model.sav')
