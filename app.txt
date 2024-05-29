from flask import Flask, request, jsonify
from model.py import Perceptron
import numpy as np

app = Flask(__name__)

perceptron = Perceptron(input_size=2)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    X = np.array(data['X'])
    d = np.array(data['d'])
    perceptron.fit(X, d)
    return jsonify({"message": "Model trained successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array(data['X'])
    predictions = [perceptron.predict(x) for x in X]
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    train_model()
    app.run(host='0.0.0.0', port=5001)