import streamlit as st
from src.train_model import load_model, predict_email
from src.db import session, Email
from datetime import datetime

vectorizer, model, le = load_model()

st.title("Spam Email Classifier")

subject = st.text_input("Email Subject")
body = st.text_area("Email Body")

# Predict button
if st.button("Predict"):
    prediction = predict_email(vectorizer, model, le, subject, body)
    st.write(f"Predicted Label: {prediction}")

    # Save email if not already in DB
    email_id = "new_" + str(hash(subject + body))
    email_record = session.query(Email).filter_by(email_id=email_id).first()
    if not email_record:
        email_record = Email(
            email_id=email_id,
            subject=subject,
            body=body,
            predicted_label=prediction,
            date=datetime.now()
        )
        session.add(email_record)
        session.commit()

    # Store email ID in session_state to track across reruns
    st.session_state["current_email_id"] = email_id
    st.session_state["prediction"] = prediction

# If there’s a current email selected, show verification options
if "current_email_id" in st.session_state:
    email_record = session.query(Email).filter_by(email_id=st.session_state["current_email_id"]).first()

    corrected_label = st.selectbox(
        "Verify or correct the label",
        options=["None", "ham", "spam"],
        index=0 if email_record.label is None else ["None", "ham", "spam"].index(email_record.label)
    )

    # Save verification button
    if st.button("Save Verification"):
        if corrected_label != "None":
            email_record.label = corrected_label
            session.commit()  # Commit to DB
            st.success(f"Label updated successfully to {corrected_label}!")
