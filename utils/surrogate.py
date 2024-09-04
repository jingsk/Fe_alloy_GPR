import pandas as pd
from sklearn.model_selection import train_test_split


class surrogate_model:
    def __init__(self, name, df):
        self.name = name
        if self.name == 'Bs':
            self.label = 'Bs (T)'
        if self.name == 'Tc':
            self.label = 'Tc (K)'
        self.df = df  # Initialize the 'df' property with a default value
        self.model = None  # Initialize the 'model' property with a default value
        self.to_scale_col = None
        self.EF_col = None
        self.original_df = None
        self.cleaned_up = False
        #self. = None,  # Initialize the 'name' property with a default value

    # # getter method 
    # def df(self): 
    #     return self.df 
      
    # # setter method 
    # def set_df(self, x): 
    #     self._age = x 

    def cleanup_df(self, drop_NaN = False, drop_col_with_NaN = True):    
        #backup
        if self.original_df is not None: 
            self.df = self.original_df.copy()
        else:
            self.original_df = self.df.copy()
        
        if drop_NaN:
            self.df = self.df.dropna()
        if drop_col_with_NaN:
            #currently hardcoded but can be changed 
            self.df = self.df.drop([
                'Annealing Time (s)',
                'Annealing Temperature (K)'
                ], axis =1)

    def split_train_test(self,test_size, seed):
        X = self.df.drop([
            self.label, 
            'composition',
            'formula',
                    ], axis =1)
        y = self.df[self.label]
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,                                                
            y.values, 
            test_size=test_size,
            random_state=seed
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        