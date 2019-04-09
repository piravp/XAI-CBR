import copy
import sklearn
import numpy as np
import lime
import lime.lime_tabular
# import string
import os
import sys

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)

def load_csv_dataset(data, target_idx, delimiter=',',
                    feature_names=None, categorical_features=None,
                    features_to_use=None, feature_transformations=None,
                    discretize=False, balance=False, fill_na='-1', filter_function=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical
    filter_function fn"""
    if feature_transformations is None: # Dont transform anything
        feature_transformations = {}
    try: # load data from text file with missing values handled as specified.
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except: # otherwise load with pandas, and fill nan values with '-1', and return store the values in data.
        import pandas
        data = pandas.read_csv(data,
                            header=None,
                            delimiter=delimiter,
                            na_filter=True,
                            dtype=str).fillna(fill_na).values
    if target_idx < 0: # if target index is set as -1, we need to find the last column.
        target_idx = data.shape[1] + target_idx # cols + (-1) = last index.
    ret = Bunch({}) # ? Create emtpy dict i think

    # If not any custom names, use the first row as features (assuming not data row)
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else: # create copy?
        feature_names = copy.deepcopy(feature_names)
    
    if skip_first:# If first row is not data, remove it.
        data = data[1:]

    if filter_function is not None: # Filter data if filter_function present
        data = filter_function(data)
    
    # Apply transformation dictionary, with corresponding transformation functions to each item (column)
    for feature, function in feature_transformations.items():
        data[:, feature] = function(data[:, feature])
    
    labels = data[:, target_idx] # Select labels from data.
    le = sklearn.preprocessing.LabelEncoder() # init label encoder
    le.fit(labels) # fit label encoder
    ret.labels = le.transform(labels) # set store transformed labels with encoder.
    labels = ret.labels # swap original labels.
    ret.class_names = list(le.classes_) # set class_names to unique label encoder classes.
    ret.class_target = feature_names[target_idx] #
    if features_to_use is not None: # check if using all features.
        data = data[:, features_to_use] # select which columns to use as features.
        feature_names = ([x for i, x in enumerate(feature_names) # Select names that we want to use
                        if i in features_to_use])
        if categorical_features is not None: # if catagorical features
            categorical_features = ([features_to_use.index(x) # select only the index of features to use.
                                    for x in categorical_features]) 
    else: # use every feature, but remove target (label)
        data = np.delete(data, target_idx, 1) # from data, delete target_idx along axix 1 (column)
        feature_names.pop(target_idx) # remove deleted column from names list
        if categorical_features: # if catagorical features
            categorical_features = ([x if x < target_idx else x - 1 # reshuffle indexes that are after removed label col
                                    for x in categorical_features]) # 0,1,2,x,4 -> 0,1,2,3 

    if categorical_features is None: # If not any catagorical features.
        categorical_features = []
        for f in range(data.shape[1]): # select every column
            if len(np.unique(data[:, f])) < 20: # if less than 20 unique values, treat as catagorical feature.
                categorical_features.append(f)
    categorical_names = {} # catagorical_names
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder() # init label encoder
        le.fit(data[:, feature]) # use column values as label encoding
        data[:, feature] = le.transform(data[:, feature]) # transform to the label
        categorical_names[feature] = le.classes_ # set categorical_names list to unique label encodings

    data = data.astype(float) # transform data to float values.
    ordinal_features = []
    if discretize: # ? turn continous values into discrete ranges (quartiles)
        disc = lime.lime_tabular.QuartileDiscretizer(data, #init discretiziser
                                                    categorical_features,
                                                    feature_names)
        data = disc.discretize(data) # discretize all data
        ordinal_features = [x for x in range(data.shape[1]) # Loop trough number of cols (features)
                            if x not in categorical_features] # resulting list of features not catagorical.
        categorical_features = range(data.shape[1]) # get list corresponding to number of catecorical features.
        categorical_names.update(disc.names) # update categorical_names with the discretized dict
    # clean categorical names.
    for x in categorical_names: # loop trough categories
        #decode numpy binary string .e.g b'string' into 'string' (utf-8) type.
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]

    ret.ordinal_features = ordinal_features # set ordinal_features (that have been discretasized)
    ret.categorical_features = categorical_features # set categorical_features
    ret.categorical_names = categorical_names # set categorical_feature_names
    ret.feature_names = feature_names # set feature_names
    np.random.seed(1) # set seed, to turn off randomization.
    if balance: # downsample the data to balance labels.
        idxs = np.array([], dtype='int') # np array of type int.
        min_labels = np.min(np.bincount(labels)) # Find the least amout of data per label.
        for label in np.unique(labels): # randomly choice datapoints to use, when too many.
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs] # keep only indexes in idxs
        labels = labels[idxs] # keep only corresonding indexes
        ret.data = data # store data in object
        ret.labels = labels # store labels in object
    # Split data, into 2 parts, training and validation 80/20 
    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                test_size=.2,
                                                random_state=1)
    # get first generated split indexes. 
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx] # train_idx split used as training data
    ret.labels_train = ret.labels[train_idx] # corresponding labels indexes data.

    # Split the remaining validation data into test and validation set. 50/50
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                    test_size=.5,
                                                    random_state=1)
    #get first generated split indexes. 
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx] # select validation indexes split
    test_idx = test_idx[ntest_idx] # select test index split

    # store datasets into bunch object. (directory object)
    ret.validation = data[cv_idx] # store validation data directory
    ret.labels_validation = ret.labels[cv_idx] # store corresponding labels.
    ret.test = data[test_idx] # store test set data
    ret.labels_test = ret.labels[test_idx] # store corresponding test labels

    ret.test_idx = test_idx # store test_indexes
    ret.validation_idx = cv_idx # store validation indexes
    ret.train_idx = train_idx # store training indexes

    # ret.train, ret.test, ret.labels_train, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(data, ret.labels,
    #                                               train_size=0.80))
    # ret.validation, ret.test, ret.labels_validation, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(ret.test, ret.labels_test,
    #                                               train_size=.5))
    ret.data = data # store all data.
    return ret # return bunch object with all data.

def load_dataset(dataset_name, balance=False, discretize=True, dataset_folder='../../Data/'):
    
    if dataset_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                        "Education-Num", "Marital Status", "Occupation",
                        "Relationship", "Race", "Sex", "Capital Gain",
                        "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
        education_map = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_map = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        }
        country_map = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_map = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        }
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}

        def cap_gains_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

        transformations = {
            3: lambda x: map_array_values(x, education_map),
            5: lambda x: map_array_values(x, married_map),
            6: lambda x: map_array_values(x, occupation_map),
            10: cap_gains_fn,
            11: cap_gains_fn,
            13: lambda x: map_array_values(x, country_map),
            14: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
    elif dataset_name == 'diabetes':
        categorical_features = [2, 3, 4, 5, 6, 7, 8, 10, 11, 18, 19, 20, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                47, 48]
        label_map = {'<30': 'YES', '>30': 'YES'}
        transformations = {
            49: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'diabetes/diabetic_data.csv'), -1, ',',
            features_to_use=range(2, 49),
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
    elif dataset_name == 'default':
        categorical_features = [2, 3, 4, 6, 7, 8, 9, 10, 11]
        dataset = load_csv_dataset(
                os.path.join(dataset_folder, 'default/default.csv'), -1, ',',
                features_to_use=range(1, 24),
                categorical_features=categorical_features, discretize=discretize,
                balance=balance)
    elif dataset_name == 'recidivism':
        features_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14]
        feature_names = ['Race', 'Alcohol', 'Junky', 'Supervised Release',
                        'Married', 'Felony', 'WorkRelease',
                        'Crime against Property', 'Crime against Person',
                        'Gender', 'Priors', 'YearsSchool', 'PrisonViolations',
                        'Age', 'MonthsServed', '', 'Recidivism']
        def violations_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'NO', '1': '1 to 5', '2': 'More than 5'})
        def priors_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [-1, 0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'UNKNOWN', '1': 'NO', '2': '1 to 5', '3': 'More than 5'})
        transformations = {
            0: lambda x: replace_binary_values(x, ['Black', 'White']),
            1: lambda x: replace_binary_values(x, ['No', 'Yes']),
            2: lambda x: replace_binary_values(x, ['No', 'Yes']),
            3: lambda x: replace_binary_values(x, ['No', 'Yes']),
            4: lambda x: replace_binary_values(x, ['No', 'Married']),
            5: lambda x: replace_binary_values(x, ['No', 'Yes']),
            6: lambda x: replace_binary_values(x, ['No', 'Yes']),
            7: lambda x: replace_binary_values(x, ['No', 'Yes']),
            8: lambda x: replace_binary_values(x, ['No', 'Yes']),
            9: lambda x: replace_binary_values(x, ['Female', 'Male']),
            10: lambda x: priors_fn(x),
            12: lambda x: violations_fn(x),
            13: lambda x: (x.astype(float) / 12).astype(int),
            16: lambda x: replace_binary_values(x, ['No more crimes',
                                                    'Re-arrested'])
        }

        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'recidivism/Data_1980.csv'), 16,
            feature_names=feature_names, discretize=discretize,
            features_to_use=features_to_use, balance=balance,
            feature_transformations=transformations, skip_first=True)
    elif dataset_name == 'lending':
        def filter_fn(data):
            to_remove = ['Does not meet the credit policy. Status:Charged Off',
            'Does not meet the credit policy. Status:Fully Paid',
            'In Grace Period', '-999', 'Current']
            for x in to_remove:
                data = data[data[:, 16] != x]
            return data
        bad_statuses = set(["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"])
        transformations = {
            16:  lambda x: np.array([y in bad_statuses for y in x]).astype(int),
            19:  lambda x: np.array([len(y) for y in x]).astype(int),
            6:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
            35:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
        }
        features_to_use = [2, 12, 13, 19, 29, 35, 51, 52, 109]
        categorical_features = [12, 109]
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'lendingclub/LoanStats3a_securev1.csv'),
            16, ',',  features_to_use=features_to_use,
            feature_transformations=transformations, fill_na='-999',
            categorical_features=categorical_features, discretize=discretize,
            filter_fn=filter_fn, balance=True)
        dataset.class_names = ['Good Loan', 'Bad Loan']

    return dataset

def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret

def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})
