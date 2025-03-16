import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from utils import get_synopsis_embeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = None
tokenizer = None


def load_model():
    """Load the SBERT model."""
    global model
    global tokenizer
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

def get_model():
    global model
    if model is None:
        load_model()
    return model

def load_data():
    """Load book data and precomputed embeddings (or generate if missing)."""
    booksDataframe = pd.read_csv("./data/books_summary.csv")

    try:
        summaries_embeddings = np.load("data/embeddings.npy")
        print("Loaded precomputed embeddings.")
    except FileNotFoundError:
        print("Embeddings not found. Encoding synopses...")
        model = get_model()
        summaries_embeddings = get_synopsis_embeddings(booksDataframe)
        np.save("data/embeddings.npy", summaries_embeddings)

    return booksDataframe, summaries_embeddings

def get_embedding(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)  # Tokenize
    max_len = 350  # Keeping space for special tokens
    overlap = 50

    
    # If text fits within limit, just encode it

    return model.encode(text, convert_to_numpy=True)
