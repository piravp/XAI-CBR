import torch
import numpy as np
from sklearn import preprocessing
from collections import defaultdict
import pandas as pd

class Datamanager():
    """ Responsible for handling data inputs to network during training, evaluation """ 
    def __init__(self, dataframe_train, dataframe_test = None,dataset=None, in_mod = 1, out_mod=1,train_frac=0.9,freq_lim=3): # 90% to be used as training, 10 % testing/val
        self.df_train = dataframe_train # training data 
        self.df_test = dataframe_test # testing data 
        self.in_mod,self.out_mod = in_mod,out_mod # decides input/output pattern

        if(self.df_test is None): # we need to divide training set into 2.
            self.train_frac = train_frac #
            self.test_frac = (1-train_frac) # 

        self.__find_limit() # find minimum number of classes.
        self.__init_data(train_frac)

        self.__pp_switcher = {
            "wine":self.wine
        }

        # format data to correct input format
        self.__i_switcher = {
            1:self.bow_stem,
            2:self.bow
            }
        
        # format data to correct output format
        self.__o_switcher = {
            1:self.one_hot_vector
        }

        #if(modus == 1):
        #    self.create_bag_of_words(stem=True,freq_lim=freq_lim)
        #elif(modus == 2):
        #    self.create_bag_of_words(freq_lim=freq_lim)

        # We need to limit the data to our limit.

    def __init_data(self,train_frac): # split dataframe into training data and validation data sets
        """ Split dataframes into datasets"""
        if(self.__pp_switcher.get(self.dataset) is None):
            raise ValueError("Couldn't preprocess dataset")

        self.training_data = [[] for x in range(2)] # list of lists [[text1,text2,text3,...],[rank1,rank2,rank3,...]]
        self.validation_data = [[] for x in range(2)] 
        # from the limit, how many of these should be for training
        train_num = self.return_num(train_frac, self.limit)
        # use limit to judge how 
        for rating in range(1,6):
            df = self.df_train.loc[self.df_train["reviewScore"] == rating] # dataframe containg only these ratings.
            if(len(df) > self.limit): # we need to reduse
                df = df.sample(frac=1) # shuffle dataframe
                df = df[:self.limit] # select first x rows 
            text_t = df["reviewText"][:train_num]
            text_v = df["reviewText"][train_num:]
            score_t = df["reviewScore"][:train_num]
            score_v = df["reviewScore"][train_num:]

            self.training_data[0].extend(text_t)
            self.training_data[1].extend(score_t)
            self.validation_data[0].extend(text_v)
            self.validation_data[1].extend(score_v)
        
        # init dataset_test
        self.testing_data = [[] for x in range(2)] # list of text + ID 
        text_t = self.df_test["reviewText"]
        id_t = self.df_test["ID"]
        self.testing_data[0].extend(text_t)
        self.testing_data[1].extend(id_t)

        print("Training set size:", len(self.training_data[0])," validation set size:",len(self.validation_data[0]))

    def __find_limit(self):
        # We need to figure out number of instances in each class, and limit each class to the lowest amount
        # dfl = df.loc[df["reviewScore"] == rating]
        count_dict = defaultdict(int)
        for rating in range(1,6):
            count_dict[rating] = len(self.df_train.loc[self.df_train["reviewScore"] == rating])
        self.limit = min(count_dict.items(),key =lambda item: item[1])[1]

    def create_bag_of_words(self,stem=False, freq_lim=3): # create dictionary mapping between word and position in array.
        # extract all words from df.
        # assumes not tokenized.
        df = self.df_train #
        raw_str = df['reviewText'].str.cat(sep=" ").split(" ")
        if(stem): # we need to itterate over the list of strings and stem everyone
            from nltk.stem.snowball import SnowballStemmer
            self.norStemmer = SnowballStemmer("norwegian")
            raw_str = [self.norStemmer.stem(s) for s in raw_str] 
        self.bow_dict = defaultdict(int) # "string":position

        # we need to remove words that occur only a few times, to reduce vocabulary
        count_dict = defaultdict(int)
        for key in raw_str:
            if(key in count_dict):
                count_dict[key] += 1
            else:
                count_dict[key] = 1
        
        stopwords = misc.get_stop_words()

        # create bag of words dictonary mapping
        for key,freq in count_dict.items():
            if(key in self.bow_dict):
                continue
            else:
                if(freq > freq_lim and key not in stopwords): # remove words than occur few times
                    self.bow_dict[key] = 1
        #then we need to figure out the length of the dictionary.
        for i,key in enumerate(self.bow_dict.keys()): # itterate every unique key
            self.bow_dict[key] = i

        self.dim = len(self.bow_dict) # input vector need to be this size to capture all words
        print("bag of words size:",self.dim)

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
        t_inputs = self.__switcher.get(self.modus)(data_inputs) # Get inputs in correct format
        # target is transformed from a number to a vector
        t_targets = torch.from_numpy(np.array([misc.int_to_one_hot_vector(int(target),5,start_val=1) for target in data_targets])).float() # cast to float tensor
        #t_targets = torch.from_numpy(np.array(data_targets)).float()
        return t_inputs, t_targets

    def return_batch(self, batch_size):
        """ Return two tensors(inputs, targets) of size batch_size from training_data"""

        #Handle converting data to correct network innput, CNN takes 3 channels.
        # Non CNN takes current setting.

        data = np.array(self.training_data) # only train on data seperate from validation_set.
        tot_size = len(self.training_data[0]) # Number of cases we got.
        if(tot_size == 0):
            raise ValueError("No training data")
        # Num, is the amount of data we send back.)
        num = self.return_num(batch_size, tot_size)
        # Randomly select num unique cases
        idx = np.random.choice(np.arange(tot_size),size=num,replace=False)
        data_inputs = data[0][idx] # split input and target
        data_targets = data[1][idx]

        #return torch.from_numpy(np.array(data_inputs)).float(), torch.from_numpy(np.array(data_targets)).float()
        return self.return_mod(data_inputs,data_targets)

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

    def return_test(self):
        data = self.testing_data 
        # as we use this only for evaluation, we don't need to shuffle. 
        if(len(data[0]) == 0):
            raise ValueError("No testing data")
        data_inputs = data[0] # split lists
        data_targets = data[1] 
        t_inputs = self.__switcher.get(self.modus)(data_inputs) # Get inputs
        return t_inputs,data_targets 

    
    # Pre processing functions

    def wine(self,data_inputs):
        # class: followed by 13 attributes as floats.
        # need to normalize
        """ Pre process wine dataset. return as [[0.1312,0.5,0.1,0.8],[0,0,1]] """
        # data inputs
        df_targets = data_inputs.iloc[:,0] # select first column
        df_attributes = data_inputs.iloc[:,1:] # select all columns expect the first

        # Normalize the data
        feature_range = len(df_attributes.columns) # get number of features
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range) # Init preprocessor.
        data_df_scaled = min_max_scaler.fit_transform(df_attributes) # Normalize Attributes.
        df_normalized = pd.DataFrame(data_df_scaled) # Transform np.array into dataframe
        
        df.targets = df.targets.apply(lambda item: misc.gen_one_hot(int(item)),classes = 3)# 

    def bow(self,data_inputs):
        """ Return vecorized sentence using bagofwords """
        if(self.bow_dict is None):
            self.create_bag_of_words()
        # we need to convert text to input vector
        inputs = []
        for string in data_inputs:
            input_vector = [0] * self.dim 

            if(isinstance(string,float)): # hack to deal with Nan fields
                inputs.append(input_vector) # add vector coresponding to text string
                continue

            for word in string.split(" "):
                if(word in self.bow_dict): # some stems arent in dictionary, since very low use
                    input_vector[self.bow_dict[word]] = 1 # change 

            inputs.append(input_vector) # add vector coresponding to text string

        return torch.from_numpy(np.array(inputs)).float() # convert list to torch tensor
    
    def bow_stem(self,data_inputs):
        """ Return vecorized stemmed sentence using bagofwords """
        if(self.bow_dict is None):
            self.create_bag_of_words(stem=True)
        # we need to convert text to input vector
        inputs = []
        for string in data_inputs:
            input_vector = [0] * self.dim 
            if(isinstance(string,float)): # hack to deal with Nan fields
                inputs.append(input_vector) # add vector coresponding to text string
                continue
            
            for stem in [self.norStemmer.stem(word) for word in string.split(" ")]:
                if(stem in self.bow_dict): # some stems arent in dictionary, since very low use
                    input_vector[self.bow_dict[stem]] = 1 # change 
            inputs.append(input_vector) # add vector coresponding to text string

        return torch.from_numpy(np.array(inputs)).float() # convert list to torch tensor

    def normalized(self,data_inputs):
        """ Return data as normalized float_value """
        pass

    def one_hot_vector(self,data_inputs):
        """ Return output as one_hot_vector """
        pass
