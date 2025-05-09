from models.stand import Model
from sklearn.svm import SVC as svc
import numpy as np
class SVC(Model):
    def __init__(self, data=None, target=None,
                 C=1, gamma=0.01, lambda_=0.01,
                 iter=100):
        super().__init__(data, target, C, gamma, lambda_, iter)
        self.model = svc(C=self.C,kernel='linear',max_iter=self.iter,decision_function_shape='ovr')

    def train(self):
        self.model.fit(self.data,self.target)
    
    def predict(self, X):
        return self.model.predict(X)

    def support_vector_ratio(self, tol=1e-3):
        w = self.model.coef_.flatten()
        return np.sum(np.abs(w) < tol) / len(w)
