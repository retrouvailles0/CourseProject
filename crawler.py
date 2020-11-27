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
import nltk
import csv


options = Options()
options.headless = True
driver = webdriver.Chrome('/usr/local/bin/chromedriver',options=options)
links = pd.read_csv("negative_link.csv")

with open(os.path.join(os.getcwd(),"negative_info.csv"), "w") as out:
    csv_w = csv.DictWriter(out, ["link", "text"], delimiter='\t')
    csv_w.writeheader()
    for url in set(links['link'].tolist()):
        if url and isinstance(url, str):
            if url.startswith("\""):
                url = url[1:-1]
            print("-"*20 + url + "-"*20)
            try:
                driver.get(url)
                res_html = driver.execute_script('return document.body.innerHTML')
                soup = BeautifulSoup(res_html,'html.parser')
                textlines = [line.strip() for line in soup.get_text().splitlines() if len(line.strip()) > 0]
                chunk = [phrase for line in textlines for phrase in line.split()]
                whole_text = ' '.join(chunk)
                csv_w.writerow({"link": url, "text": whole_text})
            except:
                continue
