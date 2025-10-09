from src.db import session, Email

recent_email = session.query(Email).order_by(Email.created_at.desc()).first()
print(recent_email.subject, recent_email.label, recent_email.predicted_label)
