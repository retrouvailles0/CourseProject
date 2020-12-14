import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup 
import sys

model = joblib.load('model.sav')

options = Options()
options.headless = True
driver = webdriver.Chrome('/usr/local/bin/chromedriver',options=options)
while True:
    print("Enter URL:")
    url = str(input())
    if url == 'exit':
        break
    driver.get(url)
    res_html = driver.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html,'html.parser')
    textlines = [line.strip() for line in soup.get_text().splitlines() if len(line.strip()) > 0]
    chunk = [phrase for line in textlines for phrase in line.split()]
    whole_text = ' '.join(chunk)
    if (model.predict([whole_text])[0] == 1):
        print("yes")
    else:
        print("no")