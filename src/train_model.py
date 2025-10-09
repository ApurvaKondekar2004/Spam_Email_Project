from src.db import session,Email
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import gensim
from gensim.models import Word2Vec 
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder

model_path="D:/Spam_Email_Project/models/word2vec.pkl"
vectorizer_path="D:/Spam_Email_Project/models/classifier.pkl"
encoder_path="D:/Spam_Email_Project/models/encoder.pkl"

def load_data():
    emails=session.query(Email).filter(Email.label.isnot(None)).all()
    df=pd.DataFrame([{"body":e.body, "label":e.label} for e in emails])
    return df

def train_word2vec(texts,vector_size=100,window=5,min_count=2):
    tokenized=[t.split() for t in texts]
    model=Word2Vec(sentences=tokenized,vector_size=vector_size,window=window,min_count=min_count,workers=4)
    return model

def vectorize_text(texts,model):
    vectors=[]
    for text in texts:
        words=text.split()
        word_vecs=[model.wv[w] for w in words if w in model.wv]
        if len(word_vecs)==0:
            vectors.append(np.zeros(model.vector_size))
        else:
            vectors.append(np.mean(word_vecs,axis=0))
    return np.array(vectors)

def train_model(data: pd.DataFrame = None):
    
    if data is None:
        data=load_data()
    if len(data)==0:
        print("No data available")
        return
    print(len(data))
    X_train,X_test,y_train,y_test=train_test_split(data["body"],data["label"],test_size=0.2,random_state=42)
    
    w2v_model=train_word2vec(X_train)
    
    X_train_vec=vectorize_text(X_train,w2v_model)
    X_test_vec=vectorize_text(X_test,w2v_model)
    
    le=LabelEncoder()
    y_train_enc=le.fit_transform(y_train)
    y_test_enc=le.transform(y_test)
    
    classifier=XGBClassifier()
    classifier.fit(X_train_vec,y_train_enc)
    
    y_pred=classifier.predict(X_test_vec)
    
    print("Accuracy:",accuracy_score(y_test_enc,y_pred))
    print(classification_report(y_test_enc,y_pred))
    
    with open(vectorizer_path,"wb") as f:
        pickle.dump(w2v_model,f)
    with open(model_path,"wb") as f:
        pickle.dump(classifier,f)
    with open(encoder_path,"wb") as f:
        pickle.dump(le,f)
    print("Model saved successfully")
    
def load_model():
    with open(vectorizer_path,"rb") as f:
        vectorizer=pickle.load(f)
    with open(model_path,"rb") as f:
        model=pickle.load(f) 
    with open(encoder_path,"rb") as f:
        le=pickle.load(f)
    return vectorizer,model,le

def predict_email(vectorizer,model,le,subject,body):
     
    from src.preprocess import clean_text
    email_text=clean_text(subject+" "+body)
     
    words=email_text.split()
    word_vecs=[vectorizer.wv[w] for w in words if w in vectorizer.wv]
    if len(word_vecs)==0:
        email_vector=np.zeros(vectorizer.vector_size).reshape(1,-1)
    else:
        email_vector=np.mean(word_vecs,axis=0).reshape(1,-1)
        
    prediction_num = model.predict(email_vector)[0]

    le.fit(["ham", "spam"])
    prediction_label = le.inverse_transform([prediction_num])[0]
    
    return prediction_label
    
if __name__=="__main__":
    train_model()
    