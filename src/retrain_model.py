from datetime import datetime, timedelta
from src.db import session, Email
from src.train_model import train_model
import pandas as pd

def retrain_model_weekly():
    # Calculate one week ago
    one_week_ago = datetime.now() - timedelta(weeks=1)
    
    # Fetch verified emails from the past week
    recent_verified_emails = session.query(Email)\
        .filter(Email.label.isnot(None))\
        .filter(Email.created_at >= one_week_ago)\
        .all()

    if not recent_verified_emails:
        print("No verified emails from this week to retrain on.")
        return

    subjects = [email.subject for email in recent_verified_emails]
    bodies = [email.body for email in recent_verified_emails]
    labels = [email.label for email in recent_verified_emails]

    texts = [f"{s} {b}" for s, b in zip(subjects, bodies)]
    data = pd.DataFrame({"body": texts, "label": labels})

    train_model(data)
    print(f"Retraining completed on {len(recent_verified_emails)} verified emails from the past week.")

if __name__ == "__main__":
    retrain_model_weekly()
