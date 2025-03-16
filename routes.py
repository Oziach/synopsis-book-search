from flask import Blueprint, request, jsonify
from model import get_model, load_data, get_embedding
from utils import compute_similarity, get_top_matches

# Create a Blueprint for routes
routes = Blueprint("routes", __name__)

# Load book data
df, synopsis_embeddings = load_data()

@routes.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    user_input = data.get("query", "").strip()  # Default to empty string if missing
    page_number = int(data.get("page_number", 1)) #default to page 1

    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    #encode user input
    model = get_model()
    user_embeddings = get_embedding(user_input).reshape(1,-1)

    #compute similarity scores
    similarity_scores = compute_similarity(user_embeddings, synopsis_embeddings)

    #get results
    results = get_top_matches(similarity_scores, df, top_n=10, page_number=page_number, score_threshold = 0.33)

    return jsonify({"results" : results})

