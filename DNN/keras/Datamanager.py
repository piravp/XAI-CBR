import numpy as np
from sklearn import preprocessing
from collections import defaultdict
import pandas as pd
import misc # helper functions

class Datamanager():
    """ Responsible for handling data inputs to network during training, evaluation and preprossesing the data """ 
    def __init__(self, classes, reduce=False, dataset=None, 
    in_mod = 1, out_mod="one-hot", train_frac=0.9,freq_lim=3): # 90% to be used as training, 10 % testing/val
        self.in_mod,self.out_mod = in_mod,out_mod # decides input/output pattern
        self.dataset = dataset
        self.classes = classes

        self.reduce = reduce 

        # Preprocess dataset using assigned function. Dataset specific
        self.__pp_switcher = {
            "wine":self.wine
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

    def return_mod(self,data_inputs,data_targets): # direct input to corresponding function
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
    
    # Pre processing functions

    def wine(self):
        """ Read whine dataset and preprosess"""
        # class: followed by 13 attributes as floats.
        self.classes=3
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
        df = read_data_pd("../../Data/wine.csv",columns = columns)

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

        return df_normalized,df_targets

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

def read_data_pd(name,columns,encoding="latin-1"):
    data = pd.read_csv(name,delimiter=",",encoding=encoding) # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe5 in position 38: invalid continuation byte
    df = pd.DataFrame(data=data) # collect panda dataframes
    return df