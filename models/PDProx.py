from stand import Model
import numpy as np

class PDProx(Model):
    def __init__(self, data=None, target=None, C=1, gamma=0.01, lambda_=0.01,iter=1000):
        super().__init__(data, target, C, gamma, lambda_,iter)
        self.w = None
        self.alpha = None

    def projection_simplex(self,v):
        if np.sum(v) <= 1 and np.all(v >= 0):
            return v
        u = np.sort(v)[::-1]
        sv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(v)+1) > (sv - 1))[0][-1]
        theta = (sv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    def update_w(self,w, G_w, gamma, lambda_):
        w_til = w - gamma * G_w 
        norm_w_til = max(np.linalg.norm(w_til, 2), 1e-10) 

        if norm_w_til == 0:
            return np.zeros_like(w) 

        w_new = w_til / max(1, gamma * lambda_ / norm_w_til)
        return w_new


    def train(self):
        """ Primal-Dual Proximal Algorithm for SVM """
        X = self.data
        y = self.target
        m, n = X.shape
        w = np.zeros(n)  
        alpha = np.zeros(m) 
        beta = np.zeros(m)  
        w_prev = np.copy(w)
        alpha_prev = np.copy(alpha)
        
        for t in range(self.iter):
            G_w = -X.T @ (alpha * y) + self.lambda_ * (w / max(np.linalg.norm(w), 1e-10)) 
            G_alpha = 1 - y * (X @ w)  
            
            alpha_new = self.projection_simplex(beta + self.gamma * G_alpha)
            
            w_new = self.update_w(w, G_w, self.gamma, self.lambda_)
            
            alpha_prev += alpha_new
            w_prev += w_new  
            w = w_new  
            alpha = alpha_new 

            beta = self.projection_simplex(beta + self.gamma * G_alpha)
        
        w = w_prev/self.iter
        alpha = alpha_prev
        self.w = w
        self.alpha = alpha

    def predict(self, X):
        y_pred = np.sign(X @ self.w)
        return y_pred
