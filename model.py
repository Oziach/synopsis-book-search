import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = None


def load_model():
    """Load the SBERT model."""
    global model
    model = SentenceTransformer(MODEL_NAME)

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
        summaries_embeddings = model.encode(booksDataframe["summaries"], convert_to_numpy=True)
        np.save("data/embeddings.npy", summaries_embeddings)

    return booksDataframe, summaries_embeddings