class BaseModel:
    """
    Base class for all models.
    Defines the interface that all models should implement.
    """
    def train(self, data, labels=None):
        raise NotImplementedError("The train method must be implemented.")

    def predict(self, data):
        raise NotImplementedError("The predict method must be implemented.")
