import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from gevent.pywsgi import WSGIServer

# Load environment variables from a .env file
load_dotenv()

# Configure the API key for google.generativeai
genai.configure(api_key=os.getenv('API_KEY'))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create a Flask application instance
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) for specific routes
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://chaemini.netlify.app",
            "https://main--chaemini.netlify.app",
            "https://chaemini-frontend.onrender.com",
        ]
    }
})

# Define the configuration for the Gemini models
config = {
    'temperature': 0,
    'top_k': 20,
    'top_p': 0.9,
    'stop_sequences': ['<|END]|>']
}

# Define safety settings for content generation
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# Initialize Gemini models without passing api_key as an argument
gemini_pro = genai.GenerativeModel('gemini-pro', generation_config=config, safety_settings=safety_settings)

# Route for generating text content
@app.route('/api/v1/g/generate-text', methods=['POST'])
def generate_text():
    # Extract prompt from request JSON data
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt or not isinstance(prompt, str):
        return jsonify({"error": "Invalid or missing 'prompt' parameter"}), 400

    # Generate text content based on the prompt using Gemini Pro model
    response = gemini_pro.generate_content(prompt)

    # Convert the response to a serializable format
    response_data = {
        'text': response.text,
        # Include any other relevant fields from the response
    }

    return jsonify(response_data)

# Initialize Gemini models without passing api_key as an argument
gemini_pro_vision = genai.GenerativeModel('gemini-pro-vision', generation_config=config, safety_settings=safety_settings)

# Route for generating vision content
@app.route('/api/v1/g/generate-vision', methods=['POST'])
def generate_vision():
    # Check if user image is included in the request files
    if 'user_image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['user_image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Read the image data and prepare prompt parts
        image_data = file.read()
        image_parts = [
            {
                "mime_type": file.content_type,
                "data": image_data
            },
        ]

        prompt_parts = [
            "Prompt for Gemini Pro Vision",
            image_parts[0],
        ]

        # Generate vision content based on the prompt using Gemini Pro Vision model
        response = gemini_pro_vision.generate_content(prompt_parts)

        return jsonify({
            "response": response.text
        })

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request"}), 400

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Route for displaying a message (e.g., a default landing page)
@app.route('/')
def display_message():
    return "This Method Isn't allowed"

if __name__ == '__main__':
    # Run the Flask application using gevent WSGIServer for production
    host = 'localhost'
    port = 5000
    print(f"App running on - http://{host}:{port}")
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()
