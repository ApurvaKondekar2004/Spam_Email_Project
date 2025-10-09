from src.etl import load_huggingface_to_db
from src.preprocess import preprocess_all
from src.train_model import train_model

if __name__=="__main__":
    
    load_huggingface_to_db()
    
    preprocess_all()
    
    train_model()