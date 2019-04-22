import numpy as np
from sklearn import preprocessing
from collections import defaultdict
import pandas as pd
import misc # helper functions
import os
import pathlib
from collections import defaultdict
import lime
class Set(object): # Set of variables.
    """ object that contain all variables we need for anchor etc"""
    def __init__(self, adict):
        self.__dict__.update(adict)

class Datamanager():
    """ Responsible for handling data inputs to network during training, evaluation and preprossesing the data """ 
    def __init__(self, reduce=False, dataset=None, 
    in_mod = 1, out_mod="one-hot", train_frac=0.9,freq_lim=3): # 90% to be used as training, 10 % testing/val
        self.in_mod,self.out_mod = in_mod,out_mod # decides input/output pattern
        self.dataset = dataset
        self.reduce = reduce 

        # Directory to store the required variables.
        self.ret = Set({}) #defaultdict() # create empty directory.

        # Preprocess dataset using assigned function. Dataset specific
        self.__pp_switcher = {
            "wine":self.wine,
            "adults":self.adults,
            "parity":self.parity
        }

        # format data to correct input format. Model specific
        self.__i_switcher = {
            1:self.normal,
        }
        
        # format data to correct output format. Model specific
        self.__o_switcher = {
            "one-hot":self.one_hot_vector,
            "float":self.float_value
        }

        #self.__find_limit() # find minimum number of classes.
        self.__init_data(train_frac)

        #if(modus == 1):
        #    self.create_bag_of_words(stem=True,freq_lim=freq_lim)
        #elif(modus == 2):
        #    self.create_bag_of_words(freq_lim=freq_lim)

        # We need to limit the data to our limit.

    def __init_data(self,train_frac): # split dataframe into training data and validation data sets
        """ Split dataframes into datasets"""
        # TODO: Handle datasets with a test set aswell
        data = self.__pp_switcher.get(self.dataset)()  # (attri, targets)
        if(data is None): 
            raise ValueError("Couldn't preprocess dataset")

        if(self.reduce):
            train_num = self.return_num(train_frac, self.reduce)
        else:
            # We want to split self.data into training and val.
            train_num = self.return_num(train_frac, len(data[0]))
        # Last column is the classes
        # to use keras.fit function, we need everything in same
        self.data_t = [data[0].values,data[1].values]

        # we need to convert dataframe into the lists.
        attributes_t = data[0][:train_num] # [0,train_num>
        attributes_v = data[0][train_num:] # [train_num, end]
        
        target_t = data[1][:train_num] # [train_num, end] 
        target_v = data[1][train_num:] # [0,train_num>

        # We need to handle multi-attributes
        self.training_data = [attributes_t.values, target_t.values]
        self.validation_data = [attributes_v.values, target_v.values]
    

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
        """ Return two tensors(inputs, targets) of size batch_size from training_data"""

        #Handle converting data to correct network innput, CNN takes 3 channels.
        # Non CNN takes current setting.
        data = self.training_data # only train on data seperate from validation_set.
        tot_size = len(self.training_data[0]) # Number of cases we got.
        if(tot_size == 0):
            raise ValueError("No training data")
        # Num, is the amount of data we send back.)
        num = self.return_num(batch_size, tot_size)
        # Randomly select num unique cases
        idx = np.random.choice(np.arange(tot_size),size=num,replace=False)
        #ndata = np.array(data[1])
        #idata = np.array(data[0])
        #print(ndata[idx])
        #print(idata[idx])
        # split input and target
        data_inputs = data[0][idx]
        data_targets = data[1][idx]
        #return torch.from_numpy(np.array(data_inputs)).float(), torch.from_numpy(np.array(data_targets)).float()
        return self.return_mod(data_inputs,data_targets)
    
    def return_keras(self): # return all data we have.
        return self.return_mod(self.data_t[0],self.data_t[1])

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
    
    #################################################
    # Pre processing functions specific to datasets #
    #   class_names: list of strings
    #   feature_names: list of strings
    #   data: used to build one hot encoder
    #   categorical_names: map from integer to list of strings, names for each
    #        value of the categorical features. Every feature that is not in
    #        this map will be considered as ordinal, and thus discretized.
    #    ordinal_features: list of integers, features that were/should be discretized.
    #################################################
    def wine(self):
        """ Read whine dataset and preprosess"""
        # class: followed by 13 attributes as floats.
        self.classes=3
        self.input_dim=13
        self.class_names = ["class 1","class 2","class 3"]
        self.feature_names = ["alch","malic","ash","alcash","mag","phen","flav","nfphens","proant","color","hue","dil","prol"]
        # need to normalize
        """ Pre process wine dataset. return as [[0.1312,0.5,0.1,0.8],[0,0,1]] """
        columns = ["class","alch","malic","ash","alcash","mag","phen","flav","nfphens","proant","color","hue","dil","prol"]
        """
            0) class
            1) Alcohol
            2) Malic acid
            3) Ash
            4) Alcalinity of ash  
            5) Magnesium
            6) Total phenols
            7) Flavanoids
            8) Nonflavanoid phenols
            9) Proanthocyanins
            10)Color intensity
            11)Hue
            12)OD280/OD315 of diluted wines
            13)Proline    
        """

        #filename = os.path.join(pathlib.Path(__file__).parents[2], "Data/wine/wine.csv")
        # Get path of parent nr 2, and append corresponding data path
        filename = pathlib.Path(__file__).parents[2]/"Data/wine/wine.csv"
        #os.path.
        df = read_data_pd(filename,columns = columns, header=False)

        df.columns = columns # Add columns to dataframe.
        #Cov.columns = ["Sequence", "Start", "End", "Coverage"]

        data_inputs = df.sample(frac=1) # shuffle dataframe
        # data inputs
        df_targets = data_inputs.iloc[:,0] # select first column
        df_attributes = data_inputs.iloc[:,1:] # select all columns expect the first
        # Normalize the data
        #feature_range = len(df_attributes.columns) # get number of features
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) # Init preprocessor.
        data_df_scaled = min_max_scaler.fit_transform(df_attributes) # Normalize Attributes.
        df_normalized = pd.DataFrame(data_df_scaled) # Transform np.array into dataframe

        #print(df_targets.head())
        #df_targets = df_targets.apply(lambda item: misc.int_to_one_hot_vector(int(item), size=self.classes,zero_offset=1))# 
        #print(df_targets.head())
        #self.data_df = pd.concat([df_normalized, df_targets], axis=1)
        # Split dataset into training and test/validation
        #print(self.data_df.head())
        #self.df_train = df

        return df_normalized, df_targets

    def adults(self):
        def strip(text):
            try:
                return text.strip()
            except AttributeError:
                return text

        columns = ["Age", "Workclass", "fnlwgt", "Education",
                        "Education-Num", "Marital Status", "Occupation",
                        "Relationship", "Race", "Sex", "Capital Gain",
                        "Capital Loss", "Hours per week", "Country", 'Income']

        #39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
        # From https://github.com/marcotcr/anchor/blob/master/anchor/utils.py
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13] # col 2 (state weighting?) and 4 (duplicate of 5), not usefull.
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13] # features that are catagorical (non-continous)
        education_map = smart_dict({ # Mapping between category (simplification)
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates','Bachelors':'Bachelors'
        })
        occupation_map = { # Mapping between category (simplification)
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        }
        country_map = { # update old name mapping.
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
        married_map = { # simplification mapping
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        }
        # Label category mapping (common notation)
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000', '<=50K.': 'Less than $50,000', '>50K.': 'More than $50,000'}

        cap_gain_map = {'0': 'None', '1': 'Low', '2': 'High'} 

        # Path to data folder.
        filename = pathlib.Path(__file__).parents[2]/"Data/adult"
        # Get both datasets (test and training)
        df = read_data_pd(filename/"adult.data",header=None, columns = columns)
        df_test = read_data_pd(name=filename/"adult.test",header=None, columns=columns)

        # Remove every row that has a '?' as its value.
        df = df[df.ne('?').all(1)] 
        df_test = df_test[df_test.ne('?').all(1)]

        #print(df["Capital Gain"])

        #print(df['Capital Loss'])

        #labels = df.values[:,-1] # last index is the target values
        #print(labels)
        """
        disc = lime.lime_tabular.EntropyDiscretizer(df.values,
                                                    categorical_features,
                                                    feature_names,
                                                    labels=labels)
        """


        #TODO: Measure how many categories we should keep
        #TODO: Fix cap_gain and cap_loss medial value discretization.

        # Select column and only keep rows with value greater than 0, and calculate the median.
        cap_gain_median = np.median(df[df['Capital Gain']>0]['Capital Gain'].values)    
        cap_loss_median = np.median(df[df['Capital Loss']>0]['Capital Loss'].values)

        print(cap_gain_median, cap_loss_median)


        def cap_gains_fn(x):
            pass

        def cap_gain_fn(x, cap_gain_median): # one value at a time
            x = np.float(x)
            d = np.digitize(x,[0,cap_gain_median, float('inf')])
            return cap_gain_map.get(d)
            
        def cap_gains_fn_old(x,cap_gain_median):
            print("CAP_GAIN",type(x))
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            print(d)
            return map_array_values(d,cap_gain_map)
            #return cap_gain_map.get(d)

        cap_gain = df['Capital Gain'].values
        #print(cap_gain)

        #cap_gains_fn_old(cap_gain, cap_gain_median)
        #cap_gain_fn.applymap
        #print(df[[3]])
        #df[[10]] = df_test.iloc[:,[10]].applymap(cap_gain_fn)
        #print(df[10])
        #exit()
        def cap_loss_fn(x):
            x = np.float(x)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            print(d)
    
            return cap_gain_map.get(d)


        transformations = { # Mapping collumns to dict maps or functions.
            3: lambda x: education_map.get(x), #3
            5: lambda x: married_map.get(x), # 5
            6: lambda x: occupation_map.get(x), # 6
            13: lambda x: country_map.get(x), # 13
            14: lambda x: label_map.get(x), # 14
        }

        # We need to find the median from

        # Transformation that must happen column wise
        transformation_c = {
            10: cap_gains_fn,
            11: cap_gains_fn,
        }
        # Apply transformation dictionary, with corresponding transformation functions to each item in column
        data = df.values
        for feature, function in transformations.items(): # Aply mapping to each element in each column.
            df[df.columns[feature]] = df.iloc[:,[feature]].applymap(function) # Apply transformer to each item in column.
            df_test[df_test.columns[feature]] = df_test.iloc[:,[feature]].applymap(function)

        for feature, function in transformation_c.items():
            pass


        # ?
        #dataset = load_csv_dataset(
        #    os.path.join(dataset_folder, 'adult/adult.data'), target_idx=-1, delimiter=', ',
        #    feature_names=feature_names, features_to_use=features_to_use,
        #    categorical_features=categorical_features, discretize=discretize,
        #    balance=balance, feature_transformations=transformations)

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
            print(np.median(x[x > 0]))
            print(d)

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
        """ Return output as one_hot_vector """
        data_targets = np.array([misc.int_to_one_hot_vector(int(item), size=self.classes, zero_offset=1) for item in data_targets])
        #data_targets = data_targets.apply(lambda item: misc.int_to_one_hot_vector(int(item), size=self.classes, zero_offset=1))# 
        return data_targets

    def float_value(self,data_targets):
        """ Return output as a float, with each class coresponding to an fraction between 0 and 1 """
        pass

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
    
    def get(self,key):
        if(super().get(key) is None):
            return key
        return super().get(key)