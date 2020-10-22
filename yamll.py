# Helper functions

def outcome_type(df, y, class_max=100):

    if df[y].nunique() <= class_max:
        res = 'classification'
    else :
        res = 'regression'

    return res

def combine_partitions(train, valid, test):

    import pandas as pd

    train['split_ind'] = 'train'
    valid['split_ind'] = 'valid'
    test['split_ind'] = 'test'

    df = train.append(valid)
    df = df.append(test)
    return df

def drop_correlated_features(df, x, correlation_cutoff = 0.95):
    
    """
    Calculates correlation values between all features and removes features that are correlated greater than or equal to the correlation_cutoff value.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 

    x : str list, optional, default=None
        Vector of feature name(s) to calculate correlation values for

    correlation_cutoff : float, optional, default=0.95
        The correlation value where features will be removed if they have a value greater than or equal to the cutoff value

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        df = drop_correlated_features(df)
    """
    
    import pandas as pd

    # Create correlation matrix
    corr_matrix = df[x].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] >= 1)]
    
    df.drop(to_drop, axis = 1, inplace = True) 
    
    return df

# Exploratory data analysis

def explore_df(df):
    
    """
    A more advanced version of describe for tabular exploratory data analysis. Inlcudes additional information such as,
    missing observations, unique observations, constant feature flagging, all_missing feature flagging, feature types & outlier
    values.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        eda = explore_df(df=df)
    """
    
    import pandas as pd
    import numpy as np
    
    ft = pd.DataFrame()
    ft['type']=df.dtypes.astype(str)
    ft['feature']=ft.index
    ft['unique']=df.nunique()
    ft['missing']= df.isnull().sum()
    ft['constant']=np.where(ft['unique']==1,1,0)
    ft['all_missing']=np.where(ft['missing']==df.shape[0],1,0)

    numeric = ft.loc[(ft['type'].str.contains('float'))]['feature']
    numeric = numeric.append(ft.loc[(ft['type'].str.contains('int'))]['feature'])
    
    categorical = ft.loc[(ft['type'].str.contains('object'))]['feature']

    # Summary statistics
    lower=df[numeric].quantile(q=0.25)
    upper=df[numeric].quantile(q=0.75)
    ft['min']=df[numeric].min()
    ft['q1']=lower
    ft['median']=df[numeric].median()
    ft['mean']=df[numeric].mean()
    ft['q3']=upper
    ft['max']=df[numeric].max()

    # Caclulate outlier values
    iqr = upper - lower
    lower=lower-(1.5*iqr)
    upper=upper+(1.5*iqr)
    ft['lower_outlier']=lower
    ft['upper_outlier']=upper
    ft['skewness']=df[numeric].skew()
    
    flag_features = get_flag_features(df)
    ft['class'] = np.where(ft['type'].str.contains('float'), 'numeric', None)
    ft['class'] = np.where(ft['type'].str.contains('int'), 'numeric', ft['class'])
    ft['class'] = np.where(ft['type'].str.contains('object'), 'categorical', ft['class'])
    ft['class'] = np.where(ft['type'].str.contains('datetime'), 'datetime', ft['class'])
    ft['class'] = np.where(ft['type'].str.contains('date'), 'datetime', ft['class'])
    ft['class'] = np.where(ft['feature'].isin(flag_features), 'flag', ft['class'])
        
    ft=ft[['feature','type','class','missing','unique','constant','all_missing','min','q1','median',
         'mean','q3','max','lower_outlier','upper_outlier','skewness']]

    ft=ft.reset_index(drop=True)
    return ft

# Data pre-processing

def partition_data(df, y = None, test_percentage = 0.2, valid_percentage = 0.1, time_based_split_feature = None,
                  seed = 1234):

    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    """
    Partitions data to create train, validation and testing sets. Splits can be done in a random stratified manner, random split
    or a time split where a feature is provided to split the data with respect to time. When creating partitions based on a time
    feature, the data is ordered according to that feature and is split according to the percentage provided in the function.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 
        
    y: str, optional, default=None
        Name of the target feature, if no value is provided random sampling will be applied
        
    test_percentage: float, optional, default=0.2
        The percentage of data to allocate to the test set
    
    valid_percentage: float, optional, default=0.1
        The percentage of data to allocate to the validation set
        
    time_based_split_feature: str, optional, defaul=None
        The feature name to use in order to split the data with respect to time. 
        
    seed : int, optional, 1234
        Random number seed for reproducable results

    Returns
    -------
    pandas df
        Returns a pandas dataframe objects for train, validation and test sets
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        train, valid, test = partition_data(df=df, y='y')
    """
    
    n_test = round(df.shape[0] * test_percentage)
    n_valid = round(df.shape[0] * valid_percentage)
    
    if y is not None:
        o_type = outcome_type(df, y)
        x = df.columns.drop(y)
    else:
        x = df.columns
        o_type = 'unsupervised'

    if time_based_split_feature is not None:
        df = df.sort_values(time_based_split_feature)
        test = df.tail(round(n_test))

        train = df.iloc[ : df.shape[0] - round(n_test)]
        valid = train.tail(round(n_valid))

        train = train.iloc[ : train.shape[0] - round(n_valid)]
    
    elif o_type == 'classification' or o_type == 'regression':
        
        if o_type == 'classfication':
            df[y] = df[y].astype('category').cat.codes
            
        x_train, x_test, y_train, y_test = train_test_split(df[x], df[y],
                                                            test_size=test_percentage, random_state=seed)
        
        train = x_train.copy()
        train[y] = y_train
        test = x_test.copy()
        test[y] = y_test
        
        #Calculate new split
        valid_percentage = n_valid / x_train.shape[0]
        x_train, x_valid, y_train, y_valid = train_test_split(train[x], train[y], 
                                                              test_size=valid_percentage, random_state=seed)
        
        train = x_train.copy()
        train[y] = y_train
        valid = x_valid.copy()
        valid[y] = y_valid
        
    else: # unsupervised
        df = df.sample(frac=1, replace=False, random_state=seed) # Shuffle data

        test = df.sample(n=n_test, replace=False, random_state=seed) 
        train = df.drop(test.index)

        valid  = train.sample(n=n_valid, replace=False, random_state=seed) 
        train = train.drop(valid.index)
    return train, valid, test

def get_flag_features(df, y=None):
    
    """
    Function used to detect flagging/indicator features.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 
        
    y: str, optional, default=None
        Name of the target feature 

    Returns
    -------
    pandas df
        Returns a pandas series object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        flag_features = get_flag_features(df=df)
    """
    
    import pandas as pd
    import numpy as np
    
    ft = pd.DataFrame()
    ft['type']=df.dtypes.astype(str)
    ft['feature']=ft.index
    ft['unique']=df.nunique()
    ft['flag'] = 0
    ft = ft.loc[ft['unique'] == 2]
    
    if y is not None:
        ft = ft.loc[ft['feature'] != y]

    flag_values = ['yes','no','true','false','y','n','0','1','-1','t','f','1.0','0.0','-1.0']

    for i in range(len(ft['feature'])):
        feat = ft.iloc[i,1]
        ft.iloc[i,3] = np.where(df[feat].astype(str).str.lower().isin(flag_values).sum() / df.shape[0] == 1, 1, 0)

    ft = ft.loc[ft['flag'] == 1]
    flag_features = ft['feature']

    return flag_features.unique()

def format_features(df, y = None):
    
    """
    Applies feature formatting to features to comply with machine learning models. Boolean features are transformed to integer 
    and objects (where applicable) are tried to be formatted as datetime. The resulting feature set should only contain float,
    integer, object and datetime type features.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 
        
    y: str, optional, default=None
        Name of the target feature

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        df = format_features(df=df)
    """
    
    import pandas as pd
    import numpy as np
    
    # Try and convert relevant datetime columns to datetime format
    df = df.apply(lambda col: pd.to_datetime(col, errors='ignore') 
              if col.dtypes == object 
              else col, 
              axis=0)

    ft = pd.DataFrame()
    ft['from_type'] = df.dtypes.astype(str)
    ft['feature'] = ft.index
    logical = ft.loc[ft['from_type'] == 'bool']['feature'].unique()
    df[logical] = df[logical].astype(int) 
    
    # Transform flagging features to integer of 1 and 0
    flag_features = get_flag_features(df)
    pos_vals = ['y','yes','t','true','1','1.0']

    for feature in flag_features:
        df[feature] = np.where(df[feature].astype(str).str.lower().isin(pos_vals), 1 , 0)
    
    return df

def get_feature_types(df, y=None, id=None):

    """
    Depends on output from explore_df function.

    Detects features of different types and returns each type in individual series objects. Feature types detected and grouped are
    numeric (float and integer), categorical (object and category), datetime (date) and flags (unique of 2 containing special
    values).

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 
    y: str, optional, default=None
        Name of the target feature
    id: str, optional, default=None
        Name of the id feature

    Returns
    -------
    pandas series
        Returns a list for numeric, categorical, datetime and flag features in that order

    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        numeric, categorical, datetime, flag = get_feature_types(df)
    """

    import pandas as pd
    import numpy as np

    ft = pd.DataFrame()
    ft['type']=df.dtypes.astype(str)
    ft['feature']=ft.index

    flag_features = get_flag_features(df)
    ft['class'] = np.where(ft['type'].str.contains('float'), 'numeric', None)
    ft['class'] = np.where(ft['type'].str.contains('int'), 'numeric', ft['class'])
    ft['class'] = np.where(ft['type'].str.contains('object'), 'categorical', ft['class'])
    ft['class'] = np.where(ft['type'].str.contains('datetime'), 'datetime', ft['class'])
    ft['class'] = np.where(ft['type'].str.contains('date'), 'datetime', ft['class'])
    ft['class'] = np.where(ft['feature'].isin(flag_features), 'flag', ft['class'])

    if y is not None:
        ft = ft.loc[ft['feature'] != y]

    if id is not None:
        ft = ft.loc[~ft['feature'].isin(id)]

    numeric = ft.loc[ft['class'] == 'numeric']['feature'].unique()
    categorical = ft.loc[ft['class'] == 'categorical']['feature'].unique()
    datetime = ft.loc[ft['class'] == 'datetime']['feature'].unique()
    flag = ft.loc[ft['class'] == 'flag']['feature'].unique()
    
    return list(numeric), list(categorical), list(datetime), list(flag)

# Feature engineering

def apply_numeric_transforms(df, x, transform_type="sqrt"):
    
    """
    Applies transformation methods to numeric features that are skewed. Transformation types include, 'sqrt' and 'log'.
    
    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 
        
    x : str list, optional, default=None
        Vector of feature name(s) to apply datetime encodings for
        
    transform_type : str, optional, default='sqrt'
        The type of transformation method to apply

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        df = apply_numeric_transforms(df=df, x='y')
        
    """
    
    import pandas as pd
    import numpy as np
    
    if transform_type == "log":
        df[x] = np.log(np.abs(df[x]+1))
    
    if transform_type == "sqrt":
        df[x] = np.sqrt(np.abs(df[x]+1))
        
    return df

def apply_datetime_encoding(df, x, training_mode=False):

    import pandas as pd
    import datetime as dt

    """
    Applies date feature encoding to datetime64 formatted features.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 
        
    x : str list, optional, default=NA
        Vector of feature name(s) to apply datetime encodings for
        
    training_mode : bool, optional, default=False
        Boolean value indicating if the encoding is being applied on the training set. Applying encodings on the training set avoids creation of certain features if the outcome is constant.

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "date": pd.date_range(start='1/1/1900', periods=4)})
        df = apply_datetime_encoding(df=df, x = 'date')
    """
    for feat in x:
        
        if training_mode == True: # Prevents creating constant features
            if df[feat].dt.year.nunique() > 1:
                df[feat+'_year'] = df[feat].dt.year
            
            if df[feat].dt.month.nunique() > 1:
                df[feat+'_month'] = df[feat].dt.month
            
            if df[feat].dt.week.nunique() > 1:
                df[feat+'_week'] = df[feat].dt.week
                
            if df[feat].dt.dayofyear.nunique() > 1:
                df[feat+'_day_of_year'] = df[feat].dt.dayofyear
                
            if  df[feat].dt.day.nunique() > 1:
                df[feat+'_day_of_month'] = df[feat].dt.day
            
            if df[feat].dt.dayofweek.nunique() > 1:
                df[feat+'_weekday'] = df[feat].dt.dayofweek
                
            if df[feat].dt.hour.nunique() > 1:
                df[feat+'_hour'] = df[feat].dt.hour
                
            if df[feat].dt.minute.nunique() > 1:
                df[feat+'_minute'] = df[feat].dt.minute
        
        else:
            df[feat+'_year'] = df[feat].dt.year
            df[feat+'_month'] = df[feat].dt.month
            df[feat+'_week'] = df[feat].dt.week
            df[feat+'_day_of_year'] = df[feat].dt.dayofyear
            df[feat+'_day_of_month'] = df[feat].dt.day
            df[feat+'_weekday'] = df[feat].dt.dayofweek
            df[feat+'_hour'] = df[feat].dt.hour
            df[feat+'_minute'] = df[feat].dt.minute
              
        del df[feat]

        return df

def map_outlier_encoding(df, x, percentile_lower_cutoff=0.01, percentile_upper_cutoff=0.99):
    
    """
    Creates a mapping table for identifying outlying values and winsorizing the outlying values by the lower and upper values.
    
    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 
        
    x : str list, optional, default=NA
        Vector of numeric feature name(s)
        
    percentile_lower_cutoff : float, optional, default=0.01
        The lower percentile value cutoff value to calculate lower boundries for
        
    percentile_upper_cutoff : float, optional, default=0.99
        The upper percentile value cutoff value to calculate upper boundries for

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": [123, 10, 5, -15], "x2":[1007,10000,1450,1236], "y": [1,1,0,1]})
        d_outlier_encodings = map_outlier_encoding(df=df, x=['x1','x2'])
    """

    import pandas as pd
    import numpy as np

    lower=df[x].quantile(q=0.25)
    upper=df[x].quantile(q=0.75)

    ft = pd.DataFrame()
    iqr = upper - lower
    lower = lower - (1.5*iqr)
    upper = upper + (1.5*iqr)
    ft['tukey_lower_outlier'] = lower
    ft['tukey_upper_outlier'] = upper
    ft['percentile_lower_outlier'] = df[x].quantile(percentile_lower_cutoff)
    ft['percentile_upper_outlier'] = df[x].quantile(percentile_upper_cutoff)
    ft['feature'] = ft.index
    ft.reset_index(inplace=True, drop=True)
    ft = ft[['feature','tukey_lower_outlier','tukey_upper_outlier','percentile_lower_outlier','percentile_upper_outlier']]

    return ft

def apply_outlier_encoding(df, mapping_table, method='percentile', tracking_flags = True):

    """
    Applies the mapping values for outliers by winsorizing the outlying values by the lower and upper values. Note that if a feature has minimal spread, i.e. lower value is the same as upper value,
    no winsorizing will be done as this can potentially cause a zero variance feature.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object 
        
    mapping_table : pandas df, required
        Pandas df object created with map_outlier_encodings
        
    method : str, optional, default='percentile'
        The applicable method to apply for winsorizing outlying values. Options are percentile, tukey
        
    tracking_flags: bool, optional, default=True
        Flags which values in a feature are outliers before winsorizing them to add additional 
        information to the model on changes that have been made to the data

    Returns
    -------
    pandas df
        Returns a pandas dataframe object

    Usage
    -----
        df = pd.DataFrame({"x1": [123, 10, 5, -15], "x2":[1007,10000,1450,1236], "y": [1,1,0,1]})
        d_outlier_encodings = map_outlier_encoding(df=df, x=['x1','x2'])
        df = apply_outlier_encodings(df=df, mapping_table=d_outlier_encodings)
    """

    import pandas as pd
    import numpy as np

    for i in range(mapping_table.shape[0]):
        feature = mapping_table.iloc[i,0]
        
        if method == 'tukey':
            lower = mapping_table.iloc[i,1]
            upper = mapping_table.iloc[i,2]
            
            if lower != upper:
                
                if tracking_flags == True: 
                    df[feature+'_flag_outlier'] = np.where(df[feature] < lower, 1, 0)
                    df[feature+'_flag_outlier'] = np.where(df[feature] > upper, 1, df[feature+'_flag_outlier'])
                
                df[feature] = np.where(df[feature] < lower, lower, df[feature])
                df[feature] = np.where(df[feature] > upper, upper, df[feature])
            
        if method == 'percentile':
            lower = mapping_table.iloc[i,3]
            upper = mapping_table.iloc[i,4] 
            
            if lower != upper:
                
                if tracking_flags == True:
                    df[feature+'_flag_outlier'] = np.where(df[feature] < lower, 1, 0)
                    df[feature+'_flag_outlier'] = np.where(df[feature] > upper, 1, df[feature+'_flag_outlier'])  
                
                df[feature] = np.where(df[feature] < lower, lower, df[feature])
                df[feature] = np.where(df[feature] > upper, upper, df[feature])
            
    return df    

def map_imputation_encoding(df, x):
    
    """
    Creates a mapping table for imputation values for both numeric and categorical features. Median encoding is used
    for numeric and the mode is used for categorical.For date and time features, they need to have been engineered before
    creating the mapping table.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object containing categorical features along with the target feature
        
    x : list, required, default=None
        Vector of feature name(s) to create imputation encodings for

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        d_imputation = map_imputation_encoding(df=df, x=['x1'], y='y')
    """
    
    import pandas as pd
    import numpy as np

    im = pd.DataFrame()
    im['type'] = df.dtypes.astype(str)
    im['feature'] = im.index
    im['missing'] = df.isnull().sum()

    # Identify logical features and convert to int
    logical = im.loc[(im['type'] == 'bool')]['feature']
    df[logical] = df[logical].astype(int)
    im['type'] = df.dtypes.astype(str) # Update table

    numeric = im.loc[im['type'].str.contains('float')]['feature']
    numeric = numeric.append(im.loc[im['type'].str.contains('int')]['feature'])

    categorical = im.loc[(im['type'] == 'object')]['feature']

    im['mode'] = df[categorical].mode().iloc[0]
    im['median'] = df[numeric].median()
    im.reset_index(drop=True, inplace=True)
    im['create_flag'] = np.where(im['missing'] > 0, 1, 0)

    im = im[['feature','median','mode','create_flag']]
    im = im.loc[im['feature'].isin(x)]
    
    return im

def apply_imputation_encoding(df, mapping_table, tracking_flags=True):

    """
    Applies a mapping table to a dataframe for imputation encodings created by the functioon map_imputation_encoding.
    Numeric features are encoding using the median and categorical features using the mode. Missing flagging features 
    can also be created.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object

    mapping_table : pandas df, required, default=NA
        Pandas dataframe object containing imputation encodings

    tracking_flags : bool, optional, default=True
        Flags which levels in a feature are missing before imputing to add additional information to the model on changes
        that have been applied to the data. Tracking features will only be created for features that had missing values in the
        mapping set. This is to avoid the creation of constant columns.

    Returns
    -------
    pandas df
        Returns a pandas dataframe object

    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        df['x1'] = np.where(df['x1'] == 'c', np.nan, df['x1'])
        d_imputation = map_imputation_encoding(df)
        df = apply_imputation_encoding(df=df, mapping_table=d_imputation, tracking_flags=True)
    """

    import pandas as pd
    import numpy as np

    categorical = df.select_dtypes(include=['object','category'])

    for feature in mapping_table['feature']:

        flag_feature = feature + '_flag_missing'
        tmp_df = mapping_table.loc[mapping_table['feature'] == feature]

        if feature in categorical.columns:
            value = tmp_df['mode']
        else:
            value = tmp_df['median']

        if tracking_flags == True:
            if tmp_df['create_flag'].item() == 1:
                df[flag_feature] = np.where(df[feature].isnull(), 1, 0)

        df[feature] = df[feature] = np.where(df[feature].isnull(), value, df[feature])

    return df

def map_categorical_interactions(x):
    
    """
    Creates a mapping table containing all possible two way interactions between features.

    Parameters
    ----------
    x : pandas series, required, default=NA
        Pandas series of categorical feature names to map interaction combinations for

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        d_categorical_interactions = map_categorical_interactions(['x1','x2'])
    """
    
    from itertools import product
    import pandas as pd
    
    ci = pd.DataFrame(list(product(x, x))) 
    ci.columns = ['base_feature','interacted_feature']
    ci = ci.loc[(ci['base_feature'] != ci['interacted_feature'])]
    return ci

def apply_categorical_interactions(df, mapping_table):
    
    import pandas as pd
    import numpy as np
    
    """
    Given a categorical interaction mapping table, applies the mappings to create interacted features on the original df.
    Interactions are simply concatenated columns.
    
    Parameters
    ----------
    df: pandas df, required, default=NA
        Pandas df object containing the base (original) data to append interactions with
    
    mapping_table : pandas df, required, default=NA
        Pandas df object created with map_categorical_interactions

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "x3":[1,10,5,7], "y": [1,1,0,1]})
        d_categorical_interactions = map_categorical_interactions(['x1','x2'])
        df = apply_categorical_interactions(df=df, mapping_table=d_categorical_interactions)
    """
    
    for i in range(len(mapping_table)):
        bf = mapping_table.iloc[i,0]
        intf = mapping_table.iloc[i,1]
        nf = 'interaction_' + bf + '_' + intf
        df[nf] = df[bf] + df[intf]

    return df

def map_categorical_encoding(df, x, y=None, max_lvls = 100, min_percent = 0.025):
        
    """
    Creates a mapping table for categorical features with various possible encoding methods to engineer categorical features with. 
    Possible encoding methods are, proportional encoding, proportional ordinal encoding, proportional one hot encoding, 
    weighted target encoding, random noise target encoding, mean weighted noise target encoding, low proportional category flag. 
    Note that the amount of dummy (one hot encoded) features are controlled by the parameter value of min_percent, setting this 
    value to 0 will cause standard one hot encoding.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object containing categorical features along with the target feature

    x : list, required, default=NA
        Vector of categorical feature name(s)

    y : str, optional, default=None
        Name of the target feature

    max_levels : int, optional, default=100
        Maximum number of unique values in the target feature to decide between a
        classification or regression problem. If the unique values are lower than the parameter value a classification
        problem is detected, else a regression problem is detected

    min_percent : float, optional, default=0.025
        The minimum proportion of data contained in each level of a categorical feature before it is flagged as a low 
        proportion level and grouped into a level ALL_OTHER with all other low proportional levels. Only applicable for
        low proportional flagging features and proportional one hot encoding. Setting this to 0 will result in normal
        one hot encoding

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        d_categorical = map_categorical_encoding(df=df, x=['x1'], y='y')
    """

    import pandas as pd
    import numpy as np

    out = []

    for feat in x:
        if y is not None: # can only do target encoding if target is provided

            target_lvls = len(df[y].unique()) # get uniques in target
            if target_lvls < max_lvls: # decide if classification or regression
                df[y] = df[y].astype('category').cat.codes

            tmp=df[[feat,y]].groupby([feat]).agg({feat:'count', y:'sum'})
            tmp['feature'] = feat
            tmp['level']=tmp.index
            tmp=tmp.reset_index(level=0, drop=True).reset_index()
            tmp.rename(columns={feat: 'count', y: 'sum'}, inplace=True)

        else:
            
            tmp=df[[feat]].groupby([feat]).agg({feat:'count'})
            tmp['feature'] = feat
            tmp['level']=tmp.index
            tmp=tmp.reset_index(level=0, drop=True).reset_index()
            tmp.rename(columns={feat: 'count'}, inplace=True)

        tmp['proportional_encode'] = tmp['count'] / tmp['count'].sum()
        tmp['flag_low_prop'] = np.where(tmp['proportional_encode'] < min_percent , 1, 0) 
        tmp = tmp.sort_values('proportional_encode', ascending=False) # order for ordinal encoding
        tmp['ordinal_encode'] = ((tmp['proportional_encode'].cumsum() - 0.5) * tmp['proportional_encode']) / tmp['proportional_encode'].sum()
        tmp['onehot'] = np.where(tmp['flag_low_prop'] == 1, 'ALL_OTHER', tmp['level'])

        if y is not None:
            noise = np.random.rand(len(tmp['level']))
            glb = tmp['sum'].sum() / tmp['count'].sum()
            lmda = 1 / (1 + np.exp((tmp['count'] - 20) / 10 * -1))
            tmp['target_encode_weighted'] = ((1 - lmda) * glb) + (lmda * tmp['sum'] / tmp['count'])
            tmp['target_encode_noise'] = (tmp['sum'] / tmp['count']) + (noise * 2 *0.01 - 0.01)
            tmp['target_encode_mean'] = (tmp['target_encode_weighted'] + tmp['target_encode_noise']) / 2
        
        if y is not None:
            tmp = tmp.drop(columns=['index', 'count','sum'])
        else:
            tmp = tmp.drop(columns=['index', 'count'])
                
        out.append(tmp)
    out = pd.concat(out)
    return out

def apply_categorical_encoding(df, mapping_table, encode_mode = 'onehot', tracking_flags = True):
    
    import pandas as pd
    import numpy as np
    
    """
    Applies a mapping table to a dataframe for categorical encodings created by the functioon map_categorical_encodings.
    After the mappings have been applied the orignal categorical features are removed from the dataframe. For more options on
    encoding types refer to the parameter section.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object containing categorical features 

    mapping_table : pandas df, required , default=NA
        Pandas dataframe object containing categorical encodings

    encode_mode : str, optional, default='onehot'
        Type of encoding to apply to a new dataframe. Possible options are proportional, ordinal, target_weight, target_noise,
        target_mean. Proportional encoding is simply the relative frequency of each category. Ordinal encoding is a transformaiton
        applied to proportional encoding to simulate the probability of a level occuring. Target weighted encoding, is target mean 
        encoding with a weighting applied to force categories with a low number of observations to the mean of the dataset. Target noise
        is target mean with random noise added. Target mean encoding is simply the mean of target weight and target noise. One hot encoding
        is dependent on the minimum percentage of data in each level set in the map_categorical_encodings function,
        and applies standard one hot encoding to a new level created.

    tracking_flags : bool, optional, default=True
        Flags which levels in a categorical feature are low proportional as defined by the minimum percentage of data in each level set
        in the map_categorical_encodings function.

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        d_categorical_encodings = map_categorical_encoding(df=df, x=['x1','x2'], y='y')
        df = apply_categorical_encoding(df = df, mapping_table = d_categorical_encodings)
    """
    
    features = ['level']

    if tracking_flags == True:
        features.append('flag_low_prop')

    if encode_mode == 'proportional':
        features.append('proportional_encode')

    if encode_mode == 'ordinal':
        features.append('ordinal_encode')

    if encode_mode == 'onehot':
        features.append('onehot')

    if encode_mode == 'target_weight':
        features.append('target_encode_weighted')

    if encode_mode == 'target_noise':
        features.append('target_encode_noise')

    if encode_mode == 'target_mean':
        features.append('target_encode_mean')

    for feature in  mapping_table['feature'].unique():
        tmp = mapping_table.loc[(mapping_table['feature'] == feature)][features]
        tmp.columns = feature + '_' + tmp.columns
        tmp_feature = tmp.columns[0] # joining key
        df = pd.merge(df, tmp, how = 'left', left_on = feature, right_on = tmp_feature)

        # Use mean if new feature levels occur to impute
        if encode_mode != 'onehot':
            new_feature = tmp.columns[1]
            df[new_feature] = df[new_feature].fillna(tmp.iloc[:, 1].mean())

        del df[tmp_feature]  
        del df[feature]
        
    feats = mapping_table['feature'].unique()
    
    if encode_mode == 'onehot':
        tmp = pd.get_dummies(df.filter(regex='_onehot', axis=1))
        df = df[df.columns.drop(list(df.filter(regex='_onehot')))]
        df = pd.concat([df, tmp], axis = 1)



    return df

def map_kmeans_encoding(df, x, clusters=5, sample_size=0.3, seed=1234):
    
    """
    Creates a mapping table for numerical features by clustering a feature and computing each observation in the features distance
    to the centroid.

    Parameters
    ----------
    df : pandas df, required, default=NA
        Pandas dataframe object

    x : list, required, default=NA
        Vector of numeric feature name(s)

    clusters : int, optional, 5
        The number of clusters to create in each feature, if the number of unique values in a feature is less than te specified 
        clusters, the unique values will be used as the number of clusters
        
    sample_size : float, optional, 0.3
        The sample size used to downsample for faster computation
        
    seed : int, optional, 1234
        Random number seed for reproducable results

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        d_kmeans_encoding = map_kmeans_encoding(df=df, x=['y'], clusters=2)
        df = apply_kmeans_encoding(df=df, mapping_table=d_kmeans_encoding)
    """

    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    import pandas as pd
    import numpy as np
    import tqdm

    out = []
    
    scaler = preprocessing.StandardScaler()
    sample_n =  round(df.shape[0] * sample_size)

    tmp_df = df[x].copy().sample(sample_n, random_state = seed)

    for feature in x:

        if tmp_df[feature].nunique() < clusters:
            n_clust = tmp_df[feature].nunique()
        else :
            n_clust = clusters

        if n_clust > 1:
            scaled = scaler.fit_transform(tmp_df[feature].values.reshape(-1, 1))
            kmeans = KMeans(n_clusters=n_clust, random_state=seed).fit(scaled.reshape(-1,1))
            centers = kmeans.cluster_centers_

            res = pd.DataFrame()
            res['cluster'] = pd.Series(range(0,(clusters)))
            res['center'] = pd.Series(np.ravel(kmeans.cluster_centers_))

            temp = pd.DataFrame()
            temp['orignal'] = tmp_df[feature]
            temp['scaled'] = scaled
            temp['cluster'] = kmeans.labels_

            temp = pd.merge(temp, res, how='left', on='cluster')

            min = temp[['orignal','cluster']].groupby(['cluster']).min().reset_index()
            max = temp[['orignal','cluster']].groupby(['cluster']).max().reset_index()
            min.columns = ['cluster', 'min']
            max.columns = ['cluster', 'max']
            temp = pd.merge(temp, min, how="left", on='cluster')
            temp = pd.merge(temp, max, how="left", on='cluster')
            temp['feature'] = feature
            res = temp[['feature','min','max','cluster','center']].drop_duplicates()
            res['center'] = res['center'] * tmp_df[feature].max()

            res['min'] = res['max'].shift()
            res.iloc[0,1] =  - 10000
            res.iloc[(n_clust-1),2] = res.iloc[(n_clust-1),2] * 1000

            out.append(res)

    out = pd.concat(out)
    out = out.reset_index()
    del out['index']
    return out

def apply_kmeans_encoding(df, mapping_table, encode_type='distance_to_center'):
    
    """
    Applies the mapping table for kmeans unsupervised features. Distance to cluster centers are calculated and are added to the 
    dataset for new features.

    Parameters
    ----------
    df : pandas df, required
        Pandas dataframe object

    mapping_table : pandas df, required 
        Pandas dataframe object containing kmeans encodings
        
    encode_type : str, optional, 'distance_to_center'
        Encoding type to apply to data. Two types are avaialbe, distance_to_center calculates how far each value is to its
        respective cluster center, cluster_mapping maps float values to discrete cluster values.

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "y": [1,1,0,1]})
        d_kmeans_encoding = map_kmeans_encoding(df=df, x=['y'], clusters=2, sample_size=0.3, seed=1234)
        df = apply_kmeans_encoding(df=df, mapping_table=d_kmeans_encoding)
    """

    import pandas as pd
    import numpy as np
    
    for feature in mapping_table['feature'].unique():
        tmp = mapping_table.loc[mapping_table['feature'] == feature]
        df['cluster'] = np.nan

        for i in range(tmp.shape[0]):
            df['cluster'] = np.where((df[feature] > tmp.iloc[i,1]) & (df[feature] <= tmp.iloc[i,2]), tmp.iloc[i,3], df['cluster'])

        df = pd.merge(df, tmp[['cluster','center']], how="left", on='cluster')
        
        if encode_type == 'distance_to_center':
            df[feature + '_kmeans_encoding'] = np.where(df['center'] == 0, df[feature] - df['center'], (df[feature] / df['center']) - 1)
            del df['center']
        
        if encode_type == 'cluster_mapping':
            df[feature + '_kmeans_encoding'] = df['cluster']
            del df['center']
        
    return df

def map_numeric_interactions(numeric_features):
    
    """
    Creates a mapping table containing all possible two way interactions between features.

    Parameters
    ----------
    numeric_features : pandas series, required
        Pandas series of numeric feature names to map interaction combinations for

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "x3":[1,10,5,7], "y": [1,1,0,1]})
        res = map_numeric_interactions(['x3','y'])
    """

    from itertools import product
    import pandas as pd

    ni = pd.DataFrame(list(product(numeric_features, numeric_features))) 
    ni.columns = ['base_feature','interacted_feature']
    ni = ni.loc[(ni['base_feature'] != ni['interacted_feature'])]
    return ni

def apply_numeric_interactions(df, mapping_table):
    
    """
    Given a numeric interaction mapping table, applies the mappings to create interacted features on the original df.
    Interactions such as + (ADD), - (SUBTRACT), / (DIVIDE) and * (MULTIPLY) are created among all possible two way combinations.

    Parameters
    ----------
    df: pandas df, required
        Pandas df object containing the base (original) data to append interactions with
    
    mapping_table : pandas df, required
        Pandas df object created with map_numeric_interactions

    Returns
    -------
    pandas df
        Returns a pandas dataframe object
    
    Usage
    -----
        df = pd.DataFrame({"x1": ["a", "b", "c", "a"], "x2":['x','y','x','x'], "x3":[1,10,5,7], "y": [1,1,0,1]})
        ni = map_numeric_interactions(['x3','y'])
        df = apply_numeric_interactions(df=df, mapping_table=ni)
    """
    
    import pandas as pd
    import numpy as np

    mapping_table.reset_index(drop = True, inplace = True)
    
    for i in range(mapping_table.shape[0]):
        feat1 = mapping_table.loc[i,"base_feature"]
        feat2 = mapping_table.loc[i,"interacted_feature"]
        
        feat = 'interaction_' + feat1 + '_ADD_' + feat2
        df[feat] = df[feat1] + df[feat2]

        feat = 'interaction_' + feat1 + '_SUBTRACT_' + feat2
        df[feat] = df[feat1] - df[feat2]

        feat = 'interaction_' + feat1 + '_MULTIPLY_' + feat2
        df[feat] = df[feat1] * df[feat2]

        feat = 'interaction_' + feat1 + '_DIVIDE_' + feat2
        df[feat] = np.where(df[feat2] == 0, df[feat1] / (df[feat2] + 1), df[feat1] / df[feat2])

    return df

# Pipeline 
def make_pipeline(pipeline_name=None, datetime_encoding=True, categorical_interactions=True, categorical_encode_mode="target_mean",
                    imputation_tracking=True, categorical_tracking=True, outlier_clipping_mode="percentile", outlier_tracking=True, outlier_percentile_lower=0.01, outlier_percentile_upper=0.99, 
                    kmeans_encoding=True,kmeans_encode_mode="distance_to_center", numeric_transform_mode="sqrt", numeric_transform_cutoff=7, seed=1234):

    """
    Creates a pipeline with settings on how to process data. Where a parameter value of None is applicable, the pipeline will not
    perform that specific operation.

    Parameters
    ----------
    pipeline_name: str, optional, None
        Name of the pipeline
    
    datetime_encoding: str, optional, True
        Should the pipeline create datetime features. Options are True or False
        
    categorical_interactions: str, optional, True
        Should the pipeline create categorical interaction features. Options are True or False
        
    categorical_encode_mode: str, optional, 'target_mean'
        How should the pipeline encode categorical features. Options are the same as specified in the function apply_categorical_encoding

    categorical_tracking: str, optional, True
        Should the pipeline create tracking features for categorical features. Options are True or False. For more information please see apply_categorical_encoding details.
    
    outlier_clipping_mode: str, optional, 'percentile'
        How should outliers be clipped. Options are None, 'percentile', 'tukey'
        
    outlier_tracking: str, optional, True
        If outliers are clipped, should the pipeline create tracking features for outliers. Options are True or False

    outlier_percentile_lower: float, optional, 0.01
        Lower percentile value to scan for outliers.

    outlier_percentile_upper: float, optional, 0.99
        Upper percentile value to scan for outliers.

    kmeans_encoding: str, optional, True
        Should kmeans features be created
        
    kmeans_encode_mode: str, optional, 'distance_to_center'
        How should the pipeline apply kmeans feature mappings. Options are None, 'distance_to_center', 'cluster_mapping'
        
    numeric_transform_mode: str, optional, 'sqrt'
        How should the pipeline transform skewed numerical features. Options are None, 'sqrt', 'log'
        
    numeric_transform_cutoff: float, optional, 7
        The cutoff value of the skewness statistic to decide which numeric features should be transformed
    
    seed : int, optional, 1234
        Random number seed for reproducable results
    
    Returns
    -------
    dict
        Returns a dictionary object
    
    Usage
    -----
        d_pipeline = create_pipeline(pipeline_name="test")

    """
    
    pl = {
        'pipeline_name':pipeline_name,
        'datetime_encoding':datetime_encoding,
        'imputation_tracking':imputation_tracking,
        'categorical_interactions':categorical_interactions,
        'categorical_encode_mode':categorical_encode_mode,
        'categorical_tracking':categorical_tracking,
        'outlier_clipping_mode':outlier_clipping_mode,
        'outlier_percentile_lower':outlier_percentile_lower,
        'outlier_percentile_upper':outlier_percentile_upper,
        'outlier_tracking':outlier_tracking,
        'kmeans_encoding':kmeans_encoding,
        'kmeans_encode_mode':kmeans_encode_mode,
        'numeric_transform_mode':numeric_transform_mode,
        'numeric_transform_cutoff':numeric_transform_cutoff,
        'seed':seed
}
    
    return pl