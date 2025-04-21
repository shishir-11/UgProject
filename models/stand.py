class Model:
    def __init__(self,data=None,target=None,C=1,gamma=0.01,lambda_=0.01,iter=1000):
        self.data = data
        self.target = target
        self.C = C
        self.gamma = gamma
        self.lambda_ = lambda_
        self.iter = iter

    def predict(self,X):
        pass

    def train(self,):
        pass
