from sklearn.linear_model import LogisticRegression
from models.stand import Model

class LogReg(Model):
    def __init__(self, data=None, target=None, C=1, gamma=0.01, lambda_=0.01, iter=100):
        super().__init__(data, target, C, gamma, lambda_, iter)
        self.model = LogisticRegression(C=self.C,max_iter=self.iter)
    
    def train(self):
        self.model.fit(X = self.data,  y=self.target)

    def predict(self,X):
        y_pred = self.model.predict(X)
        return y_pred