import sys

sys.path.append("D:/Spam_Email_Project")
from prefect import flow, task
from src.etl import load_huggingface_to_db
from src.train_model import train_model
from src.retrain_model import retrain_model_weekly

@task
def etl_task():
    load_huggingface_to_db()
    return "ETL completed"

@task
def train_task():
    train_model()
    return "Training completed"

@task
def retrain_task():
    retrain_model_weekly()
    return "Retraining completed"

@flow(name="spam_email_pipeline")
def spam_email_flow():
    etl_task()
    train_task()
    retrain_task()
