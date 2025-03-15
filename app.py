from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_model, load_data
from utils import compute_similarity, get_top_matches
from routes import routes

# Initialize Flask app
app = Flask(__name__)


# Allow same-origin access
CORS(app, resources={r"/*": {"origins": "*"}})  # "*" allows all origins

#load model
load_model()

# Register routes
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)