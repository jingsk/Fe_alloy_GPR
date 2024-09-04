import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class surrogate_model:
    def __init__(self, name, df):
        self.name = name
        if self.name == 'Bs':
            self.label = 'Bs (T)'
        if self.name == 'Tc':
            self.label = 'Tc (K)'
        self.df = df  # Initialize the 'df' property with a default value
        self.model = None  # Initialize the 'model' property with a default value
        self.original_df = None
        #self. = None,  # Initialize the 'name' property with a default value
    @property
    def X(self):
        return self.df.drop([
            self.label, 
            'composition',
            'formula',
        ], axis =1).values

    @property
    def y(self):
        return self.df[self.label].values

    @property
    def to_scale_col(self):
        X = self.df.drop([
            self.label, 
            'composition',
            'formula',
        ], axis =1)
        #index 0 is the idx but it's not present in value
        return [i for i, col in enumerate(X) if col in [
            'Annealing Time (s)',
            'Annealing Temperature (K)',
            'Thickness (mu m)']]
        #index 0 is the idx but it's not present in value
    @property
    def EF_col(self):
        X = self.df.drop([
            self.label, 
            'composition',
            'formula',
        ], axis =1)
        return [i for i in range(X.shape[1]) if i not in self.to_scale_col]
        
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

    def set_model(self, model):
        self.model = model
        
    def split_train_test(self,test_size, seed):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,                                                
            self.y, 
            test_size=test_size,
            random_state=seed
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def test_invariance(self):
        from .model import test_features_normalized
        print('Model {} is invariance to scaling non-ratio features: {}'.format(
            self.name, test_features_normalized(self.model, indices = self.to_scale_col)))
        print('Model {} is invariance to scaling elemental fractions: {}'.format(
            self.name, test_features_normalized(self.model, indices = self.EF_col)))
        
    def evaluate_train_test_fit(self):
        from sklearn.metrics import mean_squared_error    
        from .model import evaluateGP
        from scipy.stats import linregress
        
        y_train_predicted, y_train_stddev = evaluateGP(self.model, self.X_train)
        y_test_predicted, y_test_stddev = evaluateGP(self.model, self.X_test)
        
        for y_true, y_pred, l in zip([self.y_train, self.y_test],
                                     [y_train_predicted, y_test_predicted],
                                     ['train', 'test']):
        
            print(f"{l} R2 = {linregress(y_true, y_pred).rvalue**2: .03f}")
            print(f'{l} RMSE = {np.sqrt(mean_squared_error(y_true,y_pred)):.3f}')

    def plot_train_test_fit(self):
        import matplotlib.pyplot as plt
        from .model import evaluateGP
        import matplotlib.ticker as plticker
        
        y_train_predicted, y_train_stddev = evaluateGP(self.model, self.X_train)
        y_test_predicted, y_test_stddev = evaluateGP(self.model, self.X_test)
        
        fig, axs = plt.subplots(1,2, figsize=[6,3])
        
        for y_truth, y_model, stddev, l in zip([self.y_train, self.y_test],
                                          [y_train_predicted, y_test_predicted],
                                          [y_train_stddev, y_test_stddev],
                                          ['train_set', 'test_set']
                                         ):
            yerr = 2*np.array([stddev,stddev])
            markers, caps, bars = axs[0].errorbar(y_truth, y_model, yerr=yerr, label=l,
                                                  fmt='.',capsize=3, elinewidth=1, ecolor = "black")
            [bar.set_alpha(0.3) for bar in bars]
            [cap.set_alpha(0.3) for cap in caps]
            axs[1].plot(np.abs(y_truth-y_model), 2*stddev,'.', label=l)
        
        
        for ax in axs:
        
            ymin = np.min([ax.get_xlim()[0], ax.get_ylim()[0]])
            ymax = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
            ax.set_xlim([ymin, ymax])
            ax.set_ylim([ymin, ymax])
            ax.legend()
            loc = plticker.AutoLocator() # this locator puts ticks at regular intervals
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)
            ax.set_aspect('equal', 'box')
        
        if self.name == 'Bs':
            axs[0].set_xlabel(r'B$_{s}$ truth (T)')
            axs[0].set_ylabel(r'B$_{s}$ predicted (T)')
            axs[0].axline((0, 0), slope=1, color='k')
            
            axs[1].set_xlabel(r'B$_{s}$ prediction error (T)')
            axs[1].set_ylabel(r'B$_{s}$ 95% uncertainty estimate (T)')
            
        if self.name == 'Tc':
            axs[0].set_xlabel(r'T$_{c}$ truth (K)')
            axs[0].set_ylabel(r'T$_{c}$ predicted (K)')
            axs[0].axline((0, 0), slope=1, color='k')
            
            axs[1].set_xlabel(r'T$_{c}$ prediction error (K)')
            axs[1].set_ylabel(r'T$_{c}$ 95% uncertainty estimate (K)')
        
        fig.tight_layout()
        fig.savefig(f'./imgs/parity_{self.name}.png', dpi=300)