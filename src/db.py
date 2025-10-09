from sqlalchemy import create_engine, String, Integer, Column, Text, DateTime
from sqlalchemy.orm import declarative_base , sessionmaker
from datetime import datetime

Base=declarative_base()

class Email(Base):
    __tablename__='emails'
    id=Column(Integer,primary_key=True, autoincrement=True)
    email_id=Column(Integer, unique=True)
    subject=Column(String)
    date=Column(DateTime)
    body=Column(Text)
    label=Column(String(10))
    predicted_label=Column(String(10))
    created_at=Column(DateTime,default=datetime.now())
 
 
engine=create_engine('sqlite:///emails.db')
Base.metadata.create_all(engine)
Session=sessionmaker(bind=engine)
session=Session()

        