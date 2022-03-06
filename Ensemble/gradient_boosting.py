from copy import deepcopy
import numpy as np

class GradientBoosting:
    def __init__(self, base_learner, n_learners, learning_rate):
        self.learners = [deepcopy(base_learner) for _ in range(n_learners)]
        self.lr = learning_rate

    def fit(self, X, y):
        residual = y.copy()
        for learner in self.learners:
            learner.fit(X, residual)
            residual -= self.lr*learner.predict(X)

    def predict(self, X):
        preds = [learner.predict(X) for learner in self.learners]
        return np.array(preds).sum(axis=0)*self.lr
