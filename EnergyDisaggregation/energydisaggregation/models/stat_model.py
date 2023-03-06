from base import Base


class Stats(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
