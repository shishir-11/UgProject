from sklearn.datasets import load_breast_cancer, load_wine,load_digits,load_iris,make_classification, fetch_covtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LoadData:
    def __init__(self,scaler=None,dataset=''):
        self.scaler = scaler
        self.dataset = dataset
        self.data = None
        self.target = None

        if dataset:
            self.load_data()

    def load_data(self):
        dataset_loaders = {
            "breast_cancer": load_breast_cancer,
            "wine": load_wine,
            "digits": load_digits,
            "iris": load_iris,
            "make_classification": make_classification,
            "fetch_covtype": fetch_covtype,
        }

        if self.dataset in dataset_loaders:
            self.data, self.target = dataset_loaders[self.dataset](return_X_y=True)
        else:
            raise ValueError(f"Dataset '{self.dataset}' is not supported.")
        
        if self.scaler:
            if isinstance(self.scaler, StandardScaler):
                self.data = self.scaler.fit_transform(self.data)
            elif isinstance(self.scaler, MinMaxScaler):
                self.data = self.scaler.fit_transform(self.data)
            else:
                raise ValueError("Scaler should be either StandardScaler or MinMaxScaler.")
        
    def get_data(self):
        return self.data
    
    def get_target(self):
        return self.target

