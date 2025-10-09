**Project Overview**

This project is a Spam Email Detection System that classifies emails as spam or ham using Word2Vec embeddings and XGBoost classifier. It supports automated ETL, preprocessing, training, weekly retraining with verified emails, and provides a Streamlit-based interface for predictions and label verification.

**Features**

Load emails from Enron Spam Dataset into a database.

Preprocess text: clean HTML, remove stopwords, lemmatize.

Train Word2Vec + XGBoost model for spam classification.

Predict email labels via Streamlit app.

Human-in-the-loop verification of labels.

Automated ETL, training, and weekly retraining using Prefect.

**Tech Stack**

Backend & ML: Python, Gensim, Scikit-learn, XGBoost

Database: SQLite with SQLAlchemy ORM

Web App: Streamlit

Automation: Prefect workflow orchestration

Dataset: HuggingFace’s Enron Spam Dataset

**This is how it looks on Prefect Cloud**
<img width="1570" height="177" alt="image" src="https://github.com/user-attachments/assets/c6fc6f2e-d507-4cd9-bc94-75437d2555e5" />


