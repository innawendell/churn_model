import numpy as np

def fill_null_values(df, cols_with_missing):
    """
    A function to fill in missing values in the data frame with the mean value for each column.
    Params:
        df - dataframe with data
        cols_with_missing - a list with column names that contain missing values.
    Returns:
        df_copy - a dataframe with extrapolated null values.
    """
    df_copy = df.copy()
    # reset index
    df_copy = df_copy.reset_index(drop=True)
    # fill in any missing numerical values with the mean of that feature
    for col in cols_with_missing:
        df_copy.loc[:, col] = df_copy[col].fillna(df_copy[col].mean())
    
    return df_copy


def numeric_and_categorical_features(df):
    '''
    The function identifies numerical and categorical features in a dataframe.
    '''
    # they will have more than two unique values
    numerical = [col for col in df.columns if df[col].nunique() > 2]
    categorical = [col for col in df.columns if df[col].nunique() == 2]
    
    return numerical, categorical


def log_transform(df, cols):
    """
    The function performs a log transformation of the features specified in the cols (list of columns)
    """
    df_copy = df.copy()
    for col in cols:
        # if the val is 0, we need to perform log on val + 1
        df_copy[col] = np.where(df_copy[col] > 0, np.log(df_copy[col]), np.log(df_copy[col] +1 ))
    
    return df_copy


def fill_negative_vals(df, numerical_features):
    '''
    The function identifies the columns that have negative values which is a sign of noise. It replaced negative
    values with the feature mean. It allows for negative values in the "delta" columns.
    Params:
        df - dataframe
        numerical_features - a list with columns that have numerical data
    '''
    df_copy = df.copy()
    for col in numerical_features:
        # allow for negative values in delta columns
         if col.count('delta') ==0 and df_copy[df_copy[col] < 0].shape[0] > 0:
            # replace the negative values with the mean
            df_copy[col] = np.where(df_copy[col] < 0, df_copy[col].mean(), df_copy[col])
    
    return df_copy


def normalize_by_users(df, columns):
    """
    The function allows to normalize the specified columns by the number of users.
    """
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = df_copy[col] / df_copy['uniq_subs']
    
    return df_copy
