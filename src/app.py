from flask import Flask, request, jsonify
from model_evaluation import evaluate_model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt')
    if prompt is None:
        return jsonify({'error': 'No prompt provided'}), 400
    
    response = evaluate_model(prompt)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(port=5000)