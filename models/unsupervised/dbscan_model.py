from sklearn.cluster import DBSCAN
from models.base_model import BaseModel

class DBSCANModel(BaseModel):
    def __init__(self, eps=0.5, min_samples=5):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def train(self, data, labels=None):
        self.model.fit(data)

    def predict(self, data):
        return self.model.fit_predict(data)
