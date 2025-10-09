import re
import pandas as pd
from src.db import session,Email
import nltk
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn

wn.ensure_loaded()
lemmatizer=WordNetLemmatizer()
stopwords=set(stopwords.words("english"))

def clean_text(text):
    if text is None:
        return ""
    text=text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text=re.sub(r"http\S+","url",text)
    text=re.sub(r"\S+@\S+","email",text)
    text=re.sub(r"\d+","number",text)
    text=re.sub(r"[^\w\s]","",text)
    
    tokens=word_tokenize(text)
    tokens=[lemmatizer.lemmatize(w) for w in tokens if w not in stopwords]
    return " ".join(tokens)


def preprocess_all():
    emails = session.query(Email).all()
    for email in emails:
        email.body=clean_text(email.body)
    session.commit()
    print("All emails are preprocessed")
    
    
if __name__=="__main__":
    preprocess_all() 