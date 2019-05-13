import numpy as np
from sklearn import preprocessing
from collections import defaultdict
import pandas as pd
import misc # helper functions
import pathlib
from collections import defaultdict
import sklearn
#import lime
from DNN.Induction.Anchor import discretize
class Set(object): # Set of variables.
    """ object that contain all variables we need for anchor etc"""
    def __init__(self, adict):
        self.__dict__.update(adict)

    def items(self):
        return self.__dict__.items()

class Datamanager():
    """ Responsible for handling data inputs to network during training, evaluation and preprossesing the data """ 
    def __init__(self, reduce=False, dataset=None, 
    in_mod = "normal", out_mod="one-hot", train_frac=0.9,freq_lim=3): # 90% to be used as training, 10 % testing/val
        self.in_mod,self.out_mod = in_mod,out_mod # decides input/output pattern
        self.dataset = dataset
        self.reduce = reduce 

        # Directory to store the required variables.
        self.ret = Set({}) #defaultdict() # create empty directory.

        # Preprocess dataset using assigned function. Dataset specific
        self.__pp_switcher = {
            "adults":self.adults,
            "parity":self.parity
        }

        # format data to correct input format. Model specific
        self.__i_switcher = {
            "normal":self.normal,
        }
        
        # format data to correct output format. Model specific
        self.__o_switcher = {
            "one-hot":self.one_hot_vector,
            "float":self.float_value,
            "normal":self.normal
        }

        #self.__find_limit() # find minimum number of classes.
        self.__init_data(train_frac)

        #if(modus == 1):
        #    self.create_bag_of_words(stem=True,freq_lim=freq_lim)
        #elif(modus == 2):
        #    self.create_bag_of_words(freq_lim=freq_lim)

        # We need to limit the data to our limit.

    def __init_data(self,train_frac): # Init data preprocessesing.
        """ Split dataframes into datasets"""
        self.__pp_switcher.get(self.dataset)()  # (attri, targets)
        if(self.ret.data_train is None):
            raise ValueError("Couldn't preprocess dataset")
        return
        # TODO: check if we have validation and test dataset.


    def __find_limit(self,classes, class_position):
        # We need to figure out number of instances in each class, and limit each class to the lowest amount
        # dfl = df.loc[df["reviewScore"] == rating]
        #TODO: use dataframe to count distinct values in an datafram column and number of each
        if(not isinstance(class_position,str)):#check if is an ID
            raise ValueError(class_position," is not an string (dataframe column ID name)")
        count_dict = defaultdict(int)
        for rating in range(classes): # itterate every class
            pass
            #count_dict[rating] = len(self.df_train.loc[self.df_train[class_position] == rating])
        self.limit = min(count_dict.items(),key =lambda item: item[1])[1]

    def return_num(self, num, tot_size):
        """ Return number of cases we want to keep from total size. """
        # Either a fraction, "all" cases or a specific number.
        if(num != "all"):
            #check if an integer of a float.
            if(type(num) == float):  # if float
                if(num > 1):
                    num_floor = np.floor(num)
                    num = num - num_floor # only keep decimal places
                num = int(tot_size*num) # Keep only the specified fraction.

            if(tot_size < num): # if less cases than we ask for
                num = tot_size
            return num
        return tot_size # Know we want all cases.

    def return_mod(self, data_inputs, data_targets): # direct input to corresponding function
        # Depending on modus: 
        t_inputs = self.__i_switcher.get(self.in_mod)(data_inputs) # Get inputs in correct format
        # target is transformed from a number to a vector
        t_targets = self.__o_switcher.get(self.out_mod)(data_targets)
        #t_targets = torch.from_numpy(np.array([misc.int_to_one_hot_vector(int(target),5,start_val=1) for target in data_targets])).float() # cast to float tensor
        #t_targets = torch.from_numpy(np.array(data_targets)).float()
        return t_inputs, t_targets

    def return_batch(self, batch_size):
        """ Return two tensors(inputs, targets) of size batch_size from training dataset"""

        tot_size = self.ret.data_train.shape[0]

        if(batch_size <= 0): # if 0 or smaller
            raise ValueError("Batch size must be greater than 0")

        num = self.return_num(batch_size, tot_size) # return how many we should return
        idx = np.random.choice(np.arange(tot_size), size=num, replace=True) # Only want unique indexes

        data_inputs = self.ret.data_train[idx]
        data_targets = self.ret.train_labels[idx]

        return self.return_mod(data_inputs, data_targets)
    
    def return_keras(self): # return all data we have. From train
        return self.return_mod(self.ret.data_train, self.ret.train_labels)

    def return_keras_val(self):
        return self.return_mod(self.ret.data_validation, self.ret.validation_labels)

    def return_background(self, num): # return examples from training set.
        return self.validation_data[:num]

    def return_val(self):
        """ Return two tensors(inputs,targets) from validationset """
        data = self.validation_data 
        # as we use this only for evaluation, we don't need to shuffle. 
        if(len(data[0]) == 0):
            raise ValueError("No validation data")
        data_inputs = data[0]
        data_targets = data[1]
        #return torch.from_numpy(np.array(data_inputs)).float(), torch.from_numpy(np.array(data_targets)).float()
        return self.return_mod(data_inputs = data[0], data_targets=data[1])
    
    def adults(self):

        columns = ["age", "workclass", "fnlwgt", "education",
                        "education-num", "marital status", "occupation",
                        "relationship", "race", "sex", "capital gain",
                        "capital loss", "hours per week", "country", 'income']
        #39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
        # From https://github.com/marcotcr/anchor/blob/master/anchor/utils.py
        # col 2 (state weighting?) and 4 (duplicate of 5), not usefull.
        columns_to_remove = ["fnlwgt","education-num"] # column names do we not want to keep
        categorical_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11] # features that are catagorical (non-continous) after transform
        non_categorical = [0, 10] # Rest are categorical
        education_map = smart_dict({ # Mapping between category (simplification)
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates','Bachelors':'Bachelors'
        })
        occupation_map = smart_dict({ # Mapping between category (simplification)
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        })
        # Europe https://en.wikipedia.org/wiki/Northern_Europe#/media/File:Europe_subregion_map_UN_geoscheme.svg
        country_map = smart_dict({ # update old name mapping. europen countries mapped to region south, easth, west, north.
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_west', 'Germany': 'Euro_west', 'Greece': 'Euro_south',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_west', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_east', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_south', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_east',
            'Portugal': 'Euro_south', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_south', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia','Yugoslavia':'Euro_south'
        })
        married_map = smart_dict({ # simplification mapping
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        })
        # Label category mapping (common notation)
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000', '<=50K.': 'Less than $50,000', '>50K.': 'More than $50,000'}

        cap_gain_map = {'0': 'None', '1': 'Low', '2': 'High'} 

        # Path to data folder. Relative to file path.
        filename = pathlib.Path(__file__).parents[2]/"Data/adult"
        # Get both datasets (test and training)
        df = read_data_pd(filename/"adult.data",header=None, columns = columns)
        df_test = read_data_pd(name=filename/"adult.test",header=None, columns=columns)

        # Remove every row that has a '?' as its value.
        # df = df[df.ne('?').all(1)] 
        # df_test = df_test[df_test.ne('?').all(1)]

        # Remove None/Nan values from datasets.
        # df = df.dropna()
        df = df.replace('?', np.nan).dropna()
        # df_test = df_test.dropna()
        df_test = df_test.replace('?', np.nan).dropna()

        """
        disc = lime.lime_tabular.EntropyDiscretizer(df.values,
                                                    categorical_features,
                                                    feature_names,
                                                    labels=labels)
        """
        transformations = { # Mapping collumns to dict maps or functions.
            3: lambda x: education_map.get(x), #3
            5: lambda x: married_map.get(x), # 5
            6: lambda x: occupation_map.get(x), # 6
            13: lambda x: country_map.get(x), # 13
            14: lambda x: label_map.get(x), # 14
        }

        # Apply transformation dictionary, with corresponding transformation functions to each item in column
        for feature, function in transformations.items(): # Aply mapping to each element in each column.
            df[df.columns[feature]] = df.iloc[:,[feature]].applymap(function) # Apply transformer to each item in column.
            df_test[df_test.columns[feature]] = df_test.iloc[:,[feature]].applymap(function)

        # Select column and only keep rows with value greater than 0, and calculate the median.
        cap_gain_median = np.median(df[df['capital gain']>0]['capital gain'].values)
        cap_loss_median = np.median(df[df['capital loss']>0]['capital loss'].values)

        def cap_gain_fn(x, median): # one value at a time
            x = np.float(x)
            d = np.digitize(x,[0, median, float('inf')],
                            right=True).astype(str)
            d = str(d) # otherwise wont match to dictionary of strings
            return cap_gain_map.get(d)

        def cap_loss_fn(x, median):
            x = np.float(x)
            d = np.digitize(x, [0, median, float('inf')],
                            right=True).astype(str)
            d = str(d) # otherwise wont match to dictionary of strings
            return cap_gain_map.get(d)

        # Transformation that must happen column wise
        self.transformation_c = {
            10: lambda x: cap_gain_fn(x, cap_gain_median),
            11: lambda x: cap_loss_fn(x, cap_loss_median),
        }

        # perform transformation to each element in features
        for feature, function in self.transformation_c.items():
            df[df.columns[feature]] = df.iloc[:,[feature]].applymap(function)
            df_test[df_test.columns[feature]] = df_test.iloc[:,[feature]].applymap(function)

        # ? Check for inconsistencies
        # print(df.head())


        labels = df.values[:,-1] # last index is the target values
        labels_test = df_test.values[:,-1]

        le = preprocessing.LabelEncoder() # init label encoder
        le.fit(labels) # fit label encoder: targets -> encodings
        self.ret.label_encoder = le # store encoder
        self.ret.labels = le.transform(labels) # encode data set labels
        self.ret.labels_test = le.transform(labels_test) # encode test set labels

        self.ret.class_names = list(le.classes_) # set class_names to unique label encoder classes.
        #self.ret.class_target = columns[-1] # get column name of target

        # Remove useless features.
        df = df.drop(columns=columns_to_remove)
        df_test = df_test.drop(columns=columns_to_remove)
        # removed 2 columns, to preserve index for later.
        self.transformation_c = {
            8: lambda x: cap_gain_fn(x, cap_gain_median),
            9: lambda x: cap_loss_fn(x, cap_loss_median),
        }

        # Remove labels from data.
        data = df.iloc[:, 0:12] # All data excluding label/targets
        data_test = df_test.iloc[:,0:12]
        #print(data.columns)

        # Discretisize column age and hours per week.
        non_categorical = [0,10] # features that need to be discretizised.
        categorical_features = [f for f in range(data.shape[1]) if f not in non_categorical]

        # ? Display information of dataset
        #print(data.groupby('country').agg(['count','size','nunique']).stack())

        # * Discretisize non_categorical features using Lime Discretizer
        self.ret.feature_names = data.columns.values
        #print(self.ret.feature_names)
        disc = discretize.EntropyDiscretizer(data.values,
                                                    categorical_features,
                                                    self.ret.feature_names,
                                                    labels=self.ret.labels,
                                                    max_depth=3)
        
        #print(disc.discretize(data.values))
        #print(data.values[1],data.values[1].shape)# (12,), type(objects and int)
        disc_data = disc.discretize(data.values)

        # Store data to retrieve age later
        self.ret.data_test_full = data_test[['age', 'hours per week']].copy()#.values

        disc_data_test = disc.discretize(data_test.values)
        self.ret.ordinal_discretizer = disc

        # replace the data with the discretisized features from non_categorical
        for feature in non_categorical:
            data.iloc[:,feature] = disc_data[:,feature]
            data_test.iloc[:,feature] = disc_data_test[:,feature]

        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]

        # Create category mappings labels
        categorical_names = {} 
        categorical_encoders = {}
        for feature in categorical_features:
            le_f = preprocessing.LabelEncoder() # init new label encoder
            le_f.fit(data.iloc[:,feature]) # use column value as label encoder
            data.iloc[:,feature] = le_f.transform(data.iloc[:,feature].astype(str)) # get data column as string
            data_test.iloc[:,feature] = le_f.transform(data_test.iloc[:,feature].astype(str))
            #data.values[:,feature] = le_f.transform(data.values[:,feature])
            categorical_names[feature] = list(le_f.classes_)
            categorical_encoders[feature] = le_f
        categorical_names.update(disc.names) # update categorical_names with the discretized dict

        # store the label encoder
        self.ret.categorical_encoders = categorical_encoders
        # fill return Set
        self.ret.ordinal_features = ordinal_features
        self.ret.categorical_features = categorical_features
        self.ret.categorical_names = categorical_names
        self.ret.feature_names = df.columns.values

        # Split part of training_set into validation.
        np.random.seed(1) # reprodusabiliy

        # We don't bother with balancing the dataset.
        # ? Display information of the dataset
        # print(df.groupby('income').agg(['count','size','nunique']).stack())
        
        # Needed for function to work
        import sklearn.model_selection
        # Init splitter, random_state = 1 (seed)
        split = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                    test_size=0.5,
                                                    random_state=1)
        # All numpy arrays, as with float values.
        self.ret.data_train = data.values.astype(float)
        data_test = data_test.values.astype(float)

        self.ret.train_labels = self.ret.labels

        # Split test dataset into test and validation set, 50/50.
        test_idx, val_idx = [x for x in split.split(data_test)][0]
        # Select data rows by index from datasets, with corresponding labels.
        self.ret.data_test = data_test[test_idx]
        self.ret.test_labels = self.ret.labels_test[test_idx]

        self.ret.data_validation = data_test[val_idx]
        self.ret.validation_labels = self.ret.labels_test[val_idx]
        # Store indexes used.
        self.ret.test_idx = test_idx
        self.ret.validation_idx = val_idx
        self.ret.train_idx = np.array(range(data.shape[0]))
        self.classes = 2

    def parity(self): # Requre a number and .
    #num_bits, double=True
    # Parity, is simly x number of bits
        print("Using Parity...!")
    #   class_names: list of strings
    #   feature_names: list of strings
    #   data: used to build one hot encoder
    #   categorical_names: map from integer to list of strings, names for each
    #        value of the categorical features. Every feature that is not in
    #        this map will be considered as ordinal, and thus discretized.
    #    ordinal_features: list of integers, features that were/should be discretized.
        from Data import generator 
        n_bits = 10
        data = generator.gen_all_parity_cases(num_bits=n_bits,double=True)
        data = pd.DataFrame(np.array(data))
        df_attributes = data[0]
        df_targets = data[1]
        
        #   df_normalized, df_targets
        def capital_gain(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                                right=True).astype('|S128')
            #print(np.median(x[x > 0]))
            #print(d)

        self.feature_names = ["bit-"+str(i) for i in range(n_bits)]

        self.class_names = ["even","odd"] 
        #label_map = {[1,0]: "even", [0,1]: "odd"}
        self.categorical_features = [i for i in range(n_bits)] # all features are in this list
        #self.ret.class_names = ["even","odd"] 
        #   df_normalized, df_targets
        capital_gain(np.array([1,2,5,10,2,3]))
        transformations = {
            #n_bits:label_map
        }

        exit()
        return df_attributes, df_targets

    def normal(self,data_inputs):
        # we transform to np.array and to torch
        if(isinstance(data_inputs,np.ndarray)):
            return data_inputs
        return np.array(data_inputs)

    def one_hot_vector(self,data_targets):
        """ Return output as one_hot_vector. classes = 2, 0-> [0,1], 1 -> [1,0] """
        data_targets = np.array([misc.int_to_one_hot_vector(int(item), size=self.classes, zero_offset=1) for item in data_targets])
        #data_targets = data_targets.apply(lambda item: misc.int_to_one_hot_vector(int(item), size=self.classes, zero_offset=1))# 
        return data_targets

    def float_value(self,data_targets):
        """ Return output as a float, with each class coresponding to an fraction between 0 and 1 """
        pass

    #for i,v in enumerate(dataset.data_train[1]):
    #    print("{}:{}, ".format(dataset.feature_names[i],dataset.categorical_names[i][int(v)]),end="")
    def translate(self,row): # return translatet version of a list [1,2,2,1] -> ["between 20 and 30","Married","United States", etc]
        row = row.astype(int) # to index dictionary properly
        return [self.ret.categorical_names[i][v] for i,v in enumerate(row)]
    
    def translate_prediction(self, prediction): # simply return mapping between label encoding and label.
        return self.ret.class_names[prediction]

    def transform(self,row):
        # transform a row of raw data to encoded labels.
        # * discretisize the ordinal features.
        row = self.ret.ordinal_discretizer.discretize(row)
        # * perform custom transformation to each column
        for i, function in self.transformation_c.items():
            row[i] = function(row[i])
        # * perform encoding of all categories.
        for i,encoder in self.ret.categorical_encoders.items():
            # Need to transform each value to np.array of shape (x,)
            # And transform back to single element
            row[i] = encoder.transform(np.array([row[i]]))[0]
        return row

def read_data_pd(name,columns,header, encoding="latin-1"):
    # UnicodeDecodeError with 'utf-8': codec can't decode byte 0xe5, invalid continuation byte
    data = pd.read_csv(name, header=header, delimiter=",", encoding=encoding, skipinitialspace=True) 
    data.columns = columns
    return data

def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret

class smart_dict(dict):
    def __init__(self,*arg,**kw):
        super(smart_dict,self).__init__(*arg,**kw)
    
    def get(self,key): # return key if there is no value corresponding to key.
        if(super().get(key) is None):
            return key
        return super().get(key)