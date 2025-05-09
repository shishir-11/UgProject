from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data = load_breast_cancer()
X,y = data.data,data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = 2*y-1
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# X,y = make_classification(n_samples=10000, n_features=50, n_classes=2)
# y = 2*y-1
# X = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

def projection_simplex(v):
    if np.sum(v) <= 1 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v)+1) > (sv - 1))[0][-1]
    theta = (sv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)

def update_w(w, G_w, gamma, lambda_):
    w_til = w - gamma * G_w 
    norm_w_til = max(np.linalg.norm(w_til, 2), 1e-10) 

    if norm_w_til == 0:
        return np.zeros_like(w) 

    w_new = w_til / max(1, gamma * lambda_ / norm_w_til)
    return w_new

def primal_dual_prox_svm(X, y, C=1.0, gamma=0.01, lambda_=0.01, max_iter=1000):
    """ Primal-Dual Proximal Algorithm for SVM """
    m, n = X.shape
    w = np.zeros(n)  
    alpha = np.zeros(m) 
    beta = np.zeros(m)  
    w_prev = np.copy(w)
    
    for t in range(max_iter):
        G_w = -X.T @ (alpha * y) + lambda_ * (w / max(np.linalg.norm(w), 1e-10)) 
        G_alpha = 1 - y * (X @ w)  
        
        alpha = projection_simplex(beta + gamma * G_alpha)
        
        w_new = update_w(w, G_w, gamma, lambda_)
        
        w_prev += w_new  
        w = w_new  
        
        beta = projection_simplex(beta + gamma * G_alpha)
    
    return w, alpha

def grid_search():
    param_grid = {
        'gamma': [0.001, 0.01, 0.1, 1],
        'lambda_': [0.001, 0.01, 0.1, 1],
        'C':[0.001,0.01,0.1,1],
    }
    best_score = 0
    best_params = None

    for gamma in param_grid['gamma']:
        for lambda_ in param_grid['lambda_']:
            for C in param_grid['C']:
            
                w_opt, _ = primal_dual_prox_svm(X_train, y_train, gamma=gamma, lambda_=lambda_,C=C)
                y_pred = np.sign(X_test @ w_opt)
                acc = accuracy_score(y_test, y_pred)
    
                print(f"Gamma: {gamma}, Lambda: {lambda_}, C: {C}, Accuracy: {acc:.4f}")
                if acc > best_score:
                    best_score = acc
                    best_params = (gamma, lambda_,C)
    
    print("\nBest Params:", best_params, "Best Accuracy:", best_score)

# Run Grid Search
grid_search()
