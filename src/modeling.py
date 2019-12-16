import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import random
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def identify_collinear(data, correlation_threshold):
    """
    Finds collinear features based on the correlation coefficient between features. 
    For each pair of features with a correlation coefficient greather than `correlation_threshold`,
    only one of the pair is identified for removal. 
    
    Note: I recently found this code after finding an interested Feature Selector Package on 
    github and have been waiting for the opportunity to use it. Taken and adapted from 
    https://github.com/WillKoehrsen/feature-selector, who adapted it from 
    https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/


    Parameters
    --------
    correlation_threshold : float between 0 and 1
        Value of the Pearson correlation cofficient for identifying correlation features
    one_hot : boolean, default = False
        Whether to one-hot encode the features before calculating the correlation coefficients
    """
    corr_matrix = data.corr()    
    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    # Select the features with correlations above the threshold
    # Need to use the absolute value
    drop_list = [col for col in upper.columns if any(upper[col].abs() > correlation_threshold)]
    # Dataframe to hold correlated pairs
    record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

    # Iterate through the columns to drop to record pairs of correlated features
    for col in drop_list:
        # Find the correlated features
        corr_features = list(upper.index[upper[col].abs() > correlation_threshold])

        # Find the correlated values
        corr_values = list(upper[col][upper[col].abs() > correlation_threshold])
        drop_features = [col for _ in range(len(corr_features))]    

        # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                         'corr_feature': corr_features,
                                         'corr_value': corr_values})

        # Add to dataframe
        record_collinear = record_collinear.append(temp_df, ignore_index = True)
    

    print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(drop_list), correlation_threshold))
    return drop_list

def consolidate_tax_codes(df):
    df2 = df.copy()
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    column_codes = [col for col in df.columns if len(col)==6 and 
                    (col.lower()[0] == 'n' or col.lower()[0] == 'a')]
    code_list = [col[1:] for col in column_codes]
    counts = Counter(code_list)
    keep_list = [code for code, count in counts.items() if count > 1]
    consolidated_cols = []
    for code in keep_list:
        amount_col = 'a' + code
        number_col = 'n' + code
        new_name = 'mean_' + code
        df[new_name] = df[amount_col] / df[number_col]
        df[new_name].fillna(0, inplace=True)
        df.drop([amount_col, number_col], axis=1, inplace=True)
        consolidated_cols.append(new_name)
    
    column_slice = [col for col in df2.columns if col.lower() in df.columns]
    return pd.merge(df2[column_slice],df[consolidated_cols],
                   left_index=True, right_index=True)


def log_transformation(df, col, inverse=False):
    df = df.copy()
    if not inverse:
        df[col] = np.log(df[col]+1)
    else:
        df[col] = np.exp(df[col])-1
    return df




    
