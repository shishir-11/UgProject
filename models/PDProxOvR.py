import numpy as np
from stand import Model
from PDProx import PDProx

class PDProxOvr(Model):
    def __init__(self, base_cls=PDProx, data=None, target=None,
                 C=1, gamma=0.01, lambda_=0.01, iter=1000):
        super().__init__(data, target, C, gamma, lambda_, iter)
        self.base_cls = base_cls
        self.models = []
        self.classes_ = None

    def train(self):
        self.classes_ = np.unique(self.target)
        self.models = []
        for cls in self.classes_:
            y_bin = np.where(self.target == cls, 1, -1)  # Convert to binary
            model = self.base_cls(data = self.data, target=y_bin,C=1, gamma=0.01, lambda_=0.01, iter=1000)
            model.train()
            self.models.append(model)
    
    def predict(self, X):
        scores = np.array([model.w @ X.T for model in self.models])
        preds = np.argmax(scores, axis=0)
        return self.classes_[preds]
