from flask import Flask, request, jsonify
import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(np.insert(x, 0, 1))
        return self.activation_fn(z)

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.activation_fn(self.W.T.dot(x))
                self.W = self.W + self.lr * (d[i] - y) * x

app = Flask(__name__)

# Initializing a perceptron with 2 inputs
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
    app.run(host='0.0.0.0', port=8000)
