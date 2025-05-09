import numpy as np
from models.stand import Model
from models.PDProx import PDProx

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
        param_grid = {
            # 'gamma': np.linspace(0.001, 1, 10),
            # 'lambda_': np.linspace(0.001, 1, 10),
            # 'C': np.linspace(0.001,1,10)
            'gamma': [0.001, 0.01, 0.1, 1],
            'lambda_': [0.001, 0.01, 0.1, 1],
            'C':[0.001,0.01,0.1,1],
        }
        for cls in self.classes_:
            best_model = None
            best_score = 0
            y_bin = np.where(self.target == cls, 1, -1)  # One-vs-Rest label       
            for gamma in param_grid['gamma']:
                for lambda_ in param_grid['lambda_']:
                    for C in param_grid['C']:

                        model = self.base_cls(
                        data=self.data,
                        target=y_bin,
                        C=C,
                        gamma=gamma,
                        lambda_=lambda_,
                        iter=self.iter
                        )
                        model.train()
                        # Use decision values (dot product) to measure performance
                        scores = model.w @ self.data.T
                        preds = np.where(scores >= 0, 1, -1)
                        acc = np.mean(preds == y_bin)
                        
                        if acc > best_score:
                            best_score = acc
                            best_model = model
            self.models.append(best_model)

    def predict(self, X):
        scores = np.array([model.w @ X.T for model in self.models])
        preds = np.argmax(scores, axis=0)
        return self.classes_[preds]
