import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from urllib.request import urlopen
from bs4 import BeautifulSoup 
import numpy as np
import os
import sys
import time
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, ngram_range={1,2}, smooth_idf=False,
                             decode_error='ignore',  analyzer='word', lowercase=True, stop_words='english')
# links = pd.read_csv("positive_link.csv")
# print(len(set(links['Faculty Directory Homepage'])))

url = "https://www.seas.harvard.edu/electrical-engineering/people"
options = Options()
options.headless = True
driver = webdriver.Chrome('/usr/local/bin/chromedriver',options=options)
driver.get(url)
res_html = driver.execute_script('return document.body.innerHTML')
soup = BeautifulSoup(res_html,'html.parser')
print(soup)
textlines = [line.strip() for line in soup.get_text().splitlines() if len(line.strip()) > 0]
print(textlines)