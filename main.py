from dataset.loader import LoadData
from models import PDProx,PDProxOld, SVC, LogReg
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from results.metric import LoadMetric
from results.plot import Plotting
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
scaler = StandardScaler()
# dataset = 'breast_cancer'
dataset = 'make_classification'
# dataset = 'ionosphere'
loader = LoadData(scaler,dataset)
X, y = loader.data , loader.target
y = 2*y-1
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

metric_types = ['accuracy', 'precision', 'recall', 'f1']
metrics = {m: LoadMetric(type=m) for m in metric_types}
for m in metrics.values():
    m.load_metric()
# Run Grid Search
param_grid = {
    'gamma': [0.001, 0.01, 0.1, 1],
    'lambda_': [0.001, 0.01, 0.1, 1],
    'C': [0.001, 0.01, 0.1, 1],
}

# Grid search function
def grid_search(model_class, **kwargs):
    best_score = 0
    best_params = None
    for gamma in param_grid['gamma']:
        for lambda_ in param_grid['lambda_']:
            for C in param_grid['C']:
                model = model_class(X_train, y_train, gamma=gamma, lambda_=lambda_, C=C, **kwargs)
                model.train()
                y_pred = model.predict(X_test)
                score = metrics['accuracy'].get_score(y_test, y_pred)
                if score > best_score:
                    best_score = score
                    best_params = (gamma, lambda_, C)
    return best_params

results = {}
params = grid_search(PDProx.PDProx,iter=50)
pdprox = PDProx.PDProx(X_train, y_train, gamma=params[0], lambda_=params[1], C=params[2],iter=50)
pdprox.train()
y_pred = pdprox.predict(X_test)
results['PDProx'] = {m: metrics[m].get_score(y_test, y_pred) for m in metric_types}
print("Weights less than tolerance PDProx Simplex", pdprox.weight_sparsity())

accuracies = {'PDProx': [], 'SVC': [],'PDProxOld':[]}
acc = []
for i in range(10,51):
    pdprox = PDProx.PDProx(X_train, y_train, gamma=params[0], lambda_=params[1], C=params[2],iter=i)
    pdprox.train()
    y_pred = pdprox.predict(X_test)
    res = metrics['accuracy'].get_score(y_test,y_pred)
    accuracies['PDProx'].append(res)


params = grid_search(SVC.SVC,iter=50)
svc = SVC.SVC(X_train, y_train, gamma=params[0], lambda_=params[1], C=params[2],iter=50)
svc.train()
y_pred = svc.predict(X_test)
results['SVC'] = {m: metrics[m].get_score(y_test, y_pred) for m in metric_types}
for i in range(10,51):
    svc = SVC.SVC(X_train, y_train, gamma=params[0], lambda_=params[1], C=params[2],iter=i)
    svc.train()
    y_pred = svc.predict(X_test)
    res = metrics['accuracy'].get_score(y_test,y_pred)
    accuracies['SVC'].append(res)

params = grid_search(PDProxOld.PDProxOld,iter=50)
pdp = PDProxOld.PDProxOld(X_train, y_train, gamma=params[0], lambda_=params[1], C=params[2],iter=50)
pdp.train()
y_pred = pdp.predict(X_test)
results['PDProxOld'] = {m: metrics[m].get_score(y_test, y_pred) for m in metric_types}
# print(len(svc.model.support_) / len(X_train))
# print(svc.support_vector_ratio())
for i in range(10,51):
    pdp = PDProxOld.PDProxOld(X_train, y_train, gamma=params[0], lambda_=params[1], C=params[2],iter=i)
    pdp.train()
    y_pred = pdp.predict(X_test)
    res = metrics['accuracy'].get_score(y_test,y_pred)
    accuracies['PDProxOld'].append(res)

print("Weights less than tolerance PDProx box clip", pdp.weight_sparsity())



plotter = Plotting(
    accuracy={k: v['accuracy'] for k, v in results.items()},
    precision={k: v['precision'] for k, v in results.items()},
    recall={k: v['recall'] for k, v in results.items()},
    f1={k: v['f1'] for k, v in results.items()}
)
plotter.plot()
plotter.acit(res_dict=accuracies)