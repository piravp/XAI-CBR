#from keras.models import Sequential
import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv3D, Activation,Dropout
from keras import backend as K
from keras.optimizers import SGD,Adam,Adagrad,RMSprop
from keras.models import model_from_json
import pre_processing # handle preprosessing and generating data
import os
import misc
import pathlib
class Model():
    def __init__(self, name, optimizer=None, loss=None, model=None):
        self.modelpath = pathlib.Path(__file__).parent/"models"
        self.name = name
        # Check if modelpath directory exists.
        self.folder() 
        if(model is None): # we want to load from file instead.
            #modelpath_old = "DNN/keras/models/"+name+"/"+name+".json"
            if(os.path.exists(self.modelpath)): # means we want to load model.
                # We want to load model from file
                self.model = self.load_model(self.modelpath)
                weightPath = misc.find_newest_model(name)
                if(weightPath is not None and os.path.exists(weightPath)):
                    self.load_weights(weightPath)
            else:
                raise ValueError("No filepath found from model name to load from")
            
        else: # We want to create a new model.
            if(type(model) == list):
                self.model = Sequential(model) # model should be an keras model
            else:
                print("model not list")
                self.model = model
            if(optimizer is None or loss is None or name is None):
                raise ValueError("We need an optimizer, and loss function and a name for the network")
            self.optimizer = optimizer
            self.loss = loss
            self.name = name

            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        print(self.model.summary())

        #self.input_shape= (height, width, depth)
        #if(K.image_data_format()=="channels_first"):
        #    self.input_shape = (depth, height, width)
    def compile(self,**args): 
        self.model.compile(**args)

    def folder(self): # generate folder for model path
        save_dir = pathlib.Path(__file__).parent/"models"/self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def store(self,epoch):
        # Store model at specific filepath.
        save_dir = "models/"+self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir+"/"+self.name#+".json"
        if not os.path.exists(model_path):
            with open(model_path,"w") as json_file: # save model
                json_file.write(self.model.to_json()) # write model to json file

        weight_path = save_dir+"/"+ self.name + "_" + str(epoch)+".h5" # save a new network with an unique ID, name + epoch
        # we store weights too if the don't already exists
        if(not os.path.exists(weight_path)): # if path allready exists.
            self.model.save_weights(weight_path) # save weights to .h5 file

    # Load functions, we either load from file, or from start from scratch

    def load(self, filepath):
        if(os.path.isfile(filepath)): # this is the folder of the model
            # models/wine
            # weights in models/wine_{epoch}.5
            # model structure in wine.json # don't need to load this one
            # Dont need to open json file.
            self.model.compile(loss=self.loss,optimizer=self.optimizer) 

    def load_model(self,filepath):
        if(os.path.isfile(filepath)): # check if path exists.
            with open(filepath,"r") as file:
                json_data = file.read()
                return model_from_json(json_data)
        raise ValueError("modelpath is not a file:", filepath)

    def load_weights(self,filepath):
        if(os.path.isfile(filepath)): # this is the folder of the model
            self.model.load_weights(filepath) # load weights

    def train_anchor(self, data_train, train_labels, data_validation, validation_labels, 
        epochs, batch_size,verbose=1,use_gen=True):
        from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
        path = self.name+"/"+self.name+"-{val_acc:.3f}.hdf5" #+".json" # str path
        save = ModelCheckpoint(
            str(self.modelpath/path), monitor='val_acc', verbose=verbose, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        print(save.filepath )
        stop = EarlyStopping(monitor="val_loss",mode='min',patience=10,verbose=verbose)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=verbose, min_lr=1e-6, mode='min')
        
        if(use_gen):
            print(type(data_train),data_train.shape,type(train_labels))
            history = self.model.fit_generator(generator(data_train, train_labels, batch_size=100),
            validation_data=(data_validation, validation_labels),
            epochs=100, steps_per_epoch=20,callbacks=[save,stop,reduce_lr])
            #history = self.model.fit(generator(data_train, train_labels, batch_size=32), 
            #validation_data=generator(data_validation, validation_labels, batch_size=200), validation_steps=10,
            #epochs=epochs,steps_per_epoch=10,
            #callbacks=[save,stop,reduce_lr])
        else:
            history = self.model.fit(data_train, train_labels, 
            shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(data_validation, validation_labels),
            callbacks=[save,stop,reduce_lr])

        # Draw graph of training epochs.
        if(verbose > 0):
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.subplot(121)
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            plt.subplot(122)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            plt.show()
        

    def train(self, datamanager:pre_processing.Datamanager, epochs, batch_size):
        X,Y = datamanager.return_keras()# Return all data in CSV file. 
        self.model.fit(X,Y,shuffle=True,epochs=epochs,batch_size=batch_size, validation_split = 0.2) # assumes all data fit in memory.
        #self.model.train_on_batch(batch_size)

    def train_batch(self, datamanager:pre_processing.Datamanager, batch_size): # better for task requiring alot of memory
        # TODO: train with a loop
        
        X,Y = datamanager.return_batch(batch_size)# Return all data in CSV file. 
        self.model.train_on_batch(X,Y)

    def evaluate(self, data_train, train_labels, data_test, test_labels, batch_size=1000):
        score, acc = self.model.evaluate(data_train, train_labels, batch_size=batch_size)
        score_test, acc_test = self.model.evaluate(data_test, test_labels, batch_size=batch_size)
        print("loss_train {:.6f} loss_test {:6f} - acc_train {:.2f}%  acc_test {:.2f}%".format(score, score_test, acc*100, acc_test*100))



    def evaluate_old(self, datamanager:pre_processing.Datamanager, batch_size=None, steps=None):
        X,Y = datamanager.return_keras(self.input_type)
        score = self.model.evaluate(X,Y, batch_size=batch_size,steps=steps)
        print(score)
    
    def predict(self, data): # return an class as a one dimention np.array (c,)
        return self.model.predict_classes(data).flatten()

import numpy as np
def generator(x, y, batch_size):
    while True: # never ending loop
        #Randomly select batch_size number of sentences
        indx = np.random.choice(x.shape[0], batch_size, replace=False)
        yield x[indx], y[indx] # return selected data and corresponding labes


    #def predict(self,**kwargs): # return (N,1) ndarray
    #    return self.model.predict_classes(**kwargs)
    """ # TODO: Fix
    def predict(self,**kwargs): # use model to do a prediction.
        print("p_kwargs",kwargs)
        return self.model.predict(**kwargs)
    
    def predict_classes(self,**kwargs): # use model to do a prediction
        print("c_kwargs",kwargs)
        return self.model.predict_classes(**kwargs)[0]
        # We need to return int value of class (not probability mappings)
    """

sgd = SGD(lr=0.01)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)

from keras.regularizers import l1,l1_l2
#    
def CNN_50_25(name="CNN-50-25",dim=5,optimizer=adam):
    return Model(model=[
    Conv2D(data_format="channels_first",filters=3,input_shape=(3,dim,dim),kernel_size=3,padding="same",activation="relu"), # (Bx 75)
    #Conv3D(data_format="channel_first",filters=5,input_shape=(3,25),kernel_size=(1,3,3),padding=1,activation="relu"),
    Flatten(),
    Dense(50,activation="relu"), # (75 -> 50)
    Dense(dim*dim, activation="softmax") # 50 -> 25
    ], 
    optimizer=optimizer, loss="categorical_crossentropy",name=name)

def NN_adult(input_dim, output_dim, name="NN-Adult",optimizer=adam): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    return Model(model=[
        Dense(80, input_dim=input_dim, activation="relu",kernel_regularizer=l1(0.001),activity_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.5),
        Dense(60, input_dim=input_dim, activation="relu",kernel_regularizer=l1(0.001),activity_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.5),
        Dense(40, input_dim=input_dim, activation="relu",kernel_regularizer=l1(0.001),activity_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.5),
        Dense(20, activation="relu",kernel_regularizer=l1(0.001),activity_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.5),
        Dense(output_dim), # output layer
        Activation('sigmoid')
    ],
    optimizer=optimizer, loss="binary_crossentropy",name=name)

def NN_adult_1(input_dim, output_dim, name="NN-Adult-1",optimizer=adam): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    return Model(model=[
        Dense(60, input_dim=input_dim, activation="relu"),
        Dropout(0.4),
        Dense(40, input_dim=input_dim, activation="relu"),
        Dropout(0.3),
        Dense(20, input_dim=input_dim, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="relu"),
        Dropout(0.1),
        Dense(output_dim), # output layer
        Activation('sigmoid')
    ],
    optimizer=optimizer, loss="binary_crossentropy",name=name)    

def NN_adult_2(input_dim, output_dim, name="NN-Adult-2",optimizer=adam): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    return Model(model=[
        Dense(256, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.5),
        Dense(128, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.4),
        Dense(64, input_dim=input_dim, activation="relu",activity_regularizer=l1(0.001)),
        Dropout(0.3),
        Dense(32, input_dim=input_dim, activation="relu"),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dropout(0.1),
        Dense(output_dim), # output layer
        Activation('sigmoid')
    ],
    optimizer=optimizer, loss="binary_crossentropy",name=name)

def NN_adult_3(input_dim, output_dim, name="NN-Adult-3",optimizer=adam): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    return Model(model=[
        Dense(128, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.5),
        Dense(128, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.4),
        Dense(64, input_dim=input_dim, activation="relu",activity_regularizer=l1(0.001)),
        Dropout(0.3),
        Dense(32, input_dim=input_dim, activation="relu"),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dropout(0.1),
        Dense(output_dim,activation="sigmoid") # output layer
        #Activation('sigmoid)
    ],
    optimizer=optimizer, loss="binary_crossentropy",name=name)

def NN_3_20(input_dim, output_dim, name="NN-50-25",optimizer=adam): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    return Model(model=[
        Dense(20, input_dim=input_dim, activation="relu"),
        Dense(10, activation="relu"),
        Dense(output_dim), # output layer
        Activation('softmax')
    ],
    optimizer=optimizer, loss="categorical_crossentropy",name=name)

def CNN_25(name="CNN-25",dim=5):
    return Model(model=[
    Conv2D(data_format="channels_first",filters=3,input_shape=(3,dim,dim),kernel_size=3,padding="same",activation="relu"), # (Bx 75)
    #Conv3D(data_format="channel_first",filters=5,input_shape=(3,25),kernel_size=(1,3,3),padding=1,activation="relu"),
    Flatten(),
    Dense(dim*dim, activation="softmax") # 50 -> 25
    ],
    optimizer=adam, loss="categorical_crossentropy",name=name, input_type=2)

def try_training():
    dataman = Datamanager("Data/random_15000_1.csv")
    datamanager_test = Datamanager.Datamanager("Data/random_20000.csv")
    model_1.evaluate(datamanager, steps=100)
    model_1.train(datamanager=datamanager, epochs=100, batch_size=50)
    model_1.evaluate(datamanager, steps=100)
    model_1.store(epoch=0)
    
#try_training()
#model = load_model("models/CNN-50-25/CNN-50-25.json","models/CNN-50-25/CNN-50-25_0.h5",adam,"categorical_crossentropy","CNN-50-25",input_type=2)

"""
model = Sequential()
model.add(Conv2D(3,kernel_size=3,activation="relu",input_shape=(3,5,5)))
model.add(Flatten())
model.add(Dense(25,activation="softmax"))
dataset_train = Datamanager.Datamanager("Data/random_15000.csv",dim=5, modus=2)
model.compile(optimizer="adam",loss="categorical_crossentropy")
model.fit()
"""
