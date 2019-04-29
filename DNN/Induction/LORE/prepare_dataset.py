from util import *


def prepare_german_dataset(filename, path_data):

    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',')

    # Features Categorization
    columns = df.columns
    class_name = 'default'
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)

    discrete = ['installment_as_income_perc', 'present_res_since', 'credits_this_bank', 'people_under_maintenance']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete, continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    dataset = {
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset


def prepare_adult_dataset(filename, path_data,fill_na='-1'):

    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',',skipinitialspace=True)
    # clean columns
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # Remove useless columns
    df = df.drop(columns=['fnlwgt','education-num'])

    # Remove Missing Values
    for col in df.columns:
        if '?' in df[col].unique():
            # If value is row is '?' set value of ? to most frequent element in same column.
            mf_e = df[col].value_counts().index[0]
            df[col][df[col] == '?'] = mf_e

    # Features Categorization
    columns = df.columns.tolist() # get column names(keys) as list
    # put class_column in the first column index
    columns = columns[-1:] + columns[:-1]
    df = df[columns] # replace dataframe with this column shiftet one.
    class_name = 'class' # lable/target column
    possible_outcomes = list(df[class_name].unique()) # collect unique target values

    
    type_features, features_type = recognize_features_type(df, class_name)

    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=None, continuous=None)

    # Get index of the features that are not label/target (class_name)
    columns_tmp = list(columns)
    columns_tmp.remove(class_name) 
    idx_features = {i: col for i, col in enumerate(columns_tmp)} 

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values # get feature_values column
    y = df_le[class_name].values # get target value column

    dataset = { # dictionary to keep the different values.
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset


def prepare_compass_dataset(filename, path_data):

    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'
    df['class'] = df['decile_score'].apply(get_class)

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'class'
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)

    discrete = ['is_recid', 'is_violent_recid', 'two_year_recid']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                   continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    dataset = {
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset