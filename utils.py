import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(user_embedding, synopsis_embeddings):
    """Compute cosine similarity between user input and all book synopses."""
    similarity_scores = cosine_similarity(user_embedding, synopsis_embeddings).flatten()
    return similarity_scores

def get_top_matches(similarity_scores, df, top_n=10, page_number = 1, score_threshold = 0.0):
    """Return top N book recommendations based on similarity scores."""

    top_indices = np.argsort(similarity_scores)[::-1]
    top_scores = similarity_scores[top_indices]  # Get scores in sorted order

    # Filter indices based on the threshold
    valid_indices = top_indices[top_scores > score_threshold]
    total_pages = (len(valid_indices) + top_n - 1) // top_n  # Ceiling division


     # Handle out-of-bounds page requests
    if page_number > total_pages or len(valid_indices) == 0:
        return {"results": [], "total_pages": total_pages}  # Empty results if no matches

    #Page the results
    start_index = (page_number - 1) * top_n
    end_index = start_index + top_n
    paged_indices = valid_indices[start_index:end_index] 
    
    return {"results" : [
        {"book_name": df.iloc[idx]["book_name"], "summary": df.iloc[idx]["summaries"], "score": float(similarity_scores[idx])}
        for idx in paged_indices
    ], "total_pages": total_pages}