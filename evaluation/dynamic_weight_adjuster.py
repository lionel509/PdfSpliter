class DynamicWeightAdjuster:
    def __init__(self, initial_weights=None, learning_rate=0.1, decay_factor=0.99):
        """
        Initializes the dynamic weight adjuster.
        :param initial_weights: Dictionary of model weights.
        :param learning_rate: Rate of weight adjustment.
        :param decay_factor: Factor to decay weights over time.
        """
        self.weights = initial_weights if initial_weights else {}
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

    def update_weight(self, model_name, performance_score):
        """
        Updates the weight of a specific model based on its performance.
        :param model_name: Name of the model.
        :param performance_score: Performance score of the model.
        """
        if model_name not in self.weights:
            self.weights[model_name] = 1.0  # Initialize weight if not present
        
        # Adjust weight based on performance
        self.weights[model_name] = (
            self.weights[model_name] * self.decay_factor +
            self.learning_rate * performance_score
        )

    def get_weights(self):
        """
        Returns the current model weights.
        :return: Dictionary of model weights.
        """
        return self.weights
