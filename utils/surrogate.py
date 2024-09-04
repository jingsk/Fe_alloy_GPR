import pandas as pd
from sklearn.model_selection import train_test_split

class surrogate_model:
    def __init__(self, name):
        self.name = name
        if self.name == 'Bs':
            self._label = 'Bs (T)'
        if self.name == 'Tc':
            self.label = 'Tc (K)'
        self.df = None,  # Initialize the 'df' property with a default value
        self.model = None,  # Initialize the 'model' property with a default value
        self.to_scale_col = None
        self.EF_col = None
        #self. = None,  # Initialize the 'name' property with a default value

    @property
    def df(self):
        return self.df
    
    @property
    def model(self):
        return self.model

    @df.setter
    def df(self, value: pd.Dataframe):
        self.df = value
    )
    
    @model.setter
    def model(self, value):
        self.model = value
    )

    def cleanup_df(self, drop_NaN = False, drop_col_with_NaN = True):
        if not self.df:
            print("Df is not assigned.")
            return
        
        if drop_NaN:
            self.df = df.dropna()
        if drop_col_with_NaN:
            #currently hardcoded but can be changed 
            self.df = df.drop(['Annealing Time (s)',
                               'Annealing Temperature (K)'
                              ], axis =1)

    def split_train_test():
        if not self.df:
            print("Df is not assigned.")
            return
            
        X = df.drop([label, 
                     'composition',
                     'formula',
                    ], axis =1)
        y = df[label]
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,                                                
            y.values, 
            test_size=0.2,
            random_state=1
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        