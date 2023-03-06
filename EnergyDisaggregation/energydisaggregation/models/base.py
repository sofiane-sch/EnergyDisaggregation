from abc import ABC, abstractmethod
import pickle
import os


class Base(ABC):
    def __init__(self, data_dir="../data_storage", **kwargs):
        self.model = kwargs.get("model", None)
        self.data_dir = data_dir

    @abstractmethod
    def fit(self, X_train, y_train):
        """Fits the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Predicts the target values for the test data."""
        pass

    @abstractmethod
    def score(self, X_test, y_test):
        """Calculates the score of the model on the test data."""
        pass

    def save(self, filename):
        """Saves the trained model to a file."""
        dir = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        pickle.dump(self.model, open(dir, "wb"))
        print("Model saved to {}".format(dir))

    def load(self, filename):
        """Loads a trained model from a file."""
        dir = os.path.join(self.data_dir, filename)
        loaded_model = pickle.load(open(dir, "rb"))
        print("Model loaded from {}".format(dir))
        return loaded_model
