from sklearn.datasets import load_breast_cancer, load_wine,load_digits,load_iris,make_classification, fetch_covtype, fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
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
            "make_classification": make_classification,
            "ionosphere": fetch_openml
        }
        if self.dataset == 'make_classification':
            self.data, self.target = make_classification(n_samples=5000,n_features=20,n_classes=2,n_informative=5)
        elif self.dataset == "ionosphere":
            self.data, self.target = fetch_openml("ionosphere", version=1, return_X_y=True, as_frame=False,parser='liac-arff')
            le = LabelEncoder()
            self.target = le.fit_transform(self.target)
        elif self.dataset in dataset_loaders:
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

