import pandas as pd
from src.db import session, Email
from datasets import load_dataset, concatenate_datasets
from datetime import datetime

def load_huggingface_to_db():
    
    dataset=load_dataset("SetFit/enron_spam")
    
    data=concatenate_datasets([dataset["train"],dataset["test"]])
    
    skipped=0
    inserted=0
    
    for record in data:
        email_record=Email(
            email_id=record.get("message_id"),
            subject=record.get("subject"),
            date=record.get("date"),
            body=record.get("text") or datetime.now(),
            label=record.get("label_text"),
            predicted_label=None
        )
        exists = session.query(Email).filter_by(email_id=email_record.email_id).first()
        if exists:
            skipped += 1
            continue

        # Add record to session
        session.add(email_record)
        inserted += 1
    session.commit()
    print(f"ETL Completed: Hugging face dataset loaded into database")
    
if __name__=="__main__":
    load_huggingface_to_db()
    
    










