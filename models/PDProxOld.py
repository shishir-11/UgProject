from models.stand import Model
import numpy as np

class PDProxOld(Model):
    def __init__(self, data=None, target=None, C=1, gamma=0.01, lambda_=0.01, iter=1000):
        super().__init__(data, target, C, gamma, lambda_, iter)
        self.w = None
        self.alpha = None

    def box_clipping(self, v, lower=0, upper=1):
        """Project vector v onto the box [lower, upper]"""
        if upper is None:
            upper = self.C
        return np.clip(v, lower, upper)

    def update_w(self, w, G_w, gamma, lambda_):
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
            # Compute gradients
            G_w = -X.T @ (alpha * y) + self.lambda_ * (w / max(np.linalg.norm(w), 1e-10)) 
            G_alpha = 1 - y * (X @ w)  
            
            # Update alpha using box clipping (Step 4 in the algorithm)
            alpha_new = self.box_clipping(beta + self.gamma * G_alpha)
            
            # Update w (Step 5 in the algorithm)
            w_new = self.update_w(w, G_w, self.gamma, self.lambda_)
            
            # Accumulate for averaging
            alpha_prev += alpha_new
            w_prev += w_new  
            
            # Update current values
            w = w_new  
            alpha = alpha_new 
            
            # Update beta using box clipping (Step 6 in the algorithm)
            beta = self.box_clipping(beta + self.gamma * G_alpha)
        
        # Average the results
        w = w_prev/self.iter
        alpha = alpha_prev/self.iter  # Added division by iter to match algorithm's output
        self.w = w
        self.alpha = alpha

    def predict(self, X):
        y_pred = np.sign(X @ self.w)
        return y_pred

    def weight_sparsity(self, tol=1e-3):
        return np.sum(np.abs(self.w) < tol) / len(self.w)
