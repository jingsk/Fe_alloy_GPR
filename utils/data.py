import pandas as pd
from utils import add_feature
import numpy as np

def featurize_data(df,load_existing_df=False, pkl_path=None):
    if load_existing_df:
        if pkl_path == None:
            raise TypeError("please specify pkl path to load")
        return pd.read_pickle(pkl_path)
    else:    
        name = df.name
        df = add_feature.add_composition(df)
        df = add_feature.add_element_fraction(df)
        df.name = name
        dfs[df_name] = df 
        if pkl_path:
            df.to_pickle(pkl_path)
        return df

def describe_non_zero_mean(df):
    statistics = df.mask(df==0).describe().copy()
    return statistics.T[statistics.T['mean']>0].T

def subset_df_by_elements(df, elements):
    df = df.copy()
    return df[np.logical_and.reduce(tuple(((df[e]>0).values for e in elements)))]