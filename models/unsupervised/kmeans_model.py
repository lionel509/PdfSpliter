from sklearn.cluster import KMeans
from models.base_model import BaseModel

class KMeansModel(BaseModel):
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def train(self, data, labels=None):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)
