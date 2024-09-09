import numpy as np
from ax import Data
from ax.core.metric import Metric
import pandas as pd
from ax.utils.common.result import Ok
from botorch.models.gp_regression import SingleTaskGP
import torch 
from .model import evaluateGP

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #"device": torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
}

class GP_metric(Metric):

    def __init__(self, 
                 name, 
                 lower_is_better,
                 model: SingleTaskGP,
                 n_features: int, 
                 set_features: dict, 
                 feature_to_idx: dict, 
                 x_to_ER: dict):
        super().__init__(name,lower_is_better)
        self.model = model
        self.n_features = n_features
        self.set_features = set_features
        self.feature_to_idx = feature_to_idx
        self.x_to_ER = x_to_ER
    
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            mean, sem = self.evaluate(params)
            
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    # in practice, the mean and sem will be looked up based on trial metadata
                    # but for this tutorial we will calculate them
                    "mean": mean,
                    "sem": sem,
                }
            )
        return Ok(value=Data(df=pd.DataFrame.from_records(records)))

    def is_available_while_running(self) -> bool:
        return True
    
    def _features_to_X(self, features):
        X = torch.zeros((1, self.n_features), **tkwargs)
         
        for k,v in self.feature_to_idx.items():
            X[0,v] = features[k]
        return X
    
    def params_to_X(self,params,eps=1e-8):
        #print(params)
        features = self.set_features.copy()
        fixed_fraction = np.sum([self.set_features[k] for k in self.set_features if k !='t'])
        for k, v in params.items():
            features[self.x_to_ER[k]] = (v+eps)/(sum(params.values())+eps*len(params.values()))*(1-fixed_fraction)
        #print(features)
        return self._features_to_X(features)
    def evaluate(self, params) -> float:
        X = self.params_to_X(params)
        return evaluateGP(self.model, X.numpy())
