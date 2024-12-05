from sklearn.neural_network import MLPClassifier
from models.base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu'):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=42)

    def train(self, data, labels):
        self.model.fit(data, labels)

    def predict(self, data):
        return self.model.predict(data)
