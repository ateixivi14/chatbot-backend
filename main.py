from flask import Flask, jsonify

import ChatbotService as service
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/v1/chatbot", methods=['POST'])
def get_response():
    message = request.json['message']
    res = service.get_chatbot_response(message)
    response = jsonify({'response': str(res)}), 200
    return response


# Checks to see if the name of the package is the run as the main package.
if __name__ == "__main__":
    # Runs the Flask application only if the main.py file is being run.
    app.run()
