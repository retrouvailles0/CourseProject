from tkinter import *
import tkinter
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

window = tkinter.Tk()
window.geometry("670x200")
window.title("search interface")
query = tkinter.StringVar()
output_frame = tkinter.Frame(window).grid(row=1)
entry = tkinter.Entry(window, width=60, textvariable=query).grid(row=0, column=0, sticky=S + W + E + N)

tkinter.Label(output_frame, text="RESULT:", font='times 15 bold', height=3).grid(row=1, column=0, sticky=W)

out_text = Text(window, height=5, width=60, highlightbackground='black', highlightthickness=1, bd=10, wrap=WORD)
out_text.grid(row=2, columnspan=4, sticky=S + W + E + N)
out_text.tag_config('abs', font='times 16', spacing1=5, spacing2=1, spacing3=5)

model = joblib.load('model.sav')

options = Options()
options.headless = True
driver = webdriver.Chrome('/usr/local/bin/chromedriver',options=options)

def predict(query, text):
    text.delete('1.0', END)
    url = query.get()
    driver.get(url)
    res_html = driver.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html,'html.parser')
    textlines = [line.strip() for line in soup.get_text().splitlines() if len(line.strip()) > 0]
    chunk = [phrase for line in textlines for phrase in line.split()]
    whole_text = ' '.join(chunk)
    if (model.predict([whole_text])[0] == 1):
        text.insert(END, 'yes', 'abs')
    else:
        text.insert(END, 'no', 'abs')
button = tkinter.Button(window, text="check", fg="red", width=8, command=lambda : predict(query, out_text)).grid(row=0, column=1, sticky=S + W + E + N)
window.mainloop()