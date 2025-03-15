from flask import Flask, request, jsonify
from model import load_model, load_data
from utils import compute_similarity, get_top_matches
from routes import routes

# Initialize Flask app
app = Flask(__name__)

#load model
load_model()

# Register routes
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)