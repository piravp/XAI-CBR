
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv3D, Activation,Dropout
from keras import backend as K
from keras.optimizers import SGD,Adam,Adagrad,RMSprop
from keras.models import model_from_json
import pre_processing # handle preprosessing and generating data
import os
import misc
import pathlib
import sklearn

class BlackBox():
    def __init__(self, name, optimizer=None, loss=None, model=None,c_path=None,verbose=False):
        self.modelpath = pathlib.Path(__file__).parent/"models" # Keep track of folder path of model. 
        self.name = name
        # checkpoint callback variables.
        self.best_val_acc = 0
        self.epoch = 0

        # Check if modelpath directory exists.
        self.folder() 
        if(model is None): # we want to load from file instead.
            #modelpath_old = "DNN/keras/models/"+name+"/"+name+".json"
            if(os.path.exists(str(self.modelpath/c_path))): # means we want to load model.
                if(c_path is None): # We simply want to load the best model availible.
                    pass
                else:
                    self.load_model_checkpoint(str(self.modelpath/c_path))                        
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
        if(verbose):
            print(self.model.summary())

    def compile(self,**args): 
        self.model.compile(**args)

    def folder(self): # generate folder for model path
        save_dir = pathlib.Path(__file__).parent/"models"/self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def store(self,epoch):
        # Store model at specific filepath.
        save_dir = "models/"+self.name
        if not os.path.exists(save_dir): # check if save directory exists
            os.makedirs(save_dir)
        model_path = save_dir+"/"+self.name#+".json"
        if not os.path.exists(model_path): # check if model path already exists.
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

    def load_model_checkpoint(self,filepath):
        # load model based on checkpoint.
        if(os.path.isfile(filepath)):
            self.model = keras.models.load_model(filepath)
            #self.model.compile(loss=self.loss, optimizer=self.optimizer)
        else: # if the filepath does not exists.
            raise ValueError("filepath is not a file",filepath)

    def load_weights(self,filepath):
        if(os.path.isfile(filepath)): # this is the folder of the model
            self.model.load_weights(filepath) # load weights

    def checkPoint(self,epoch,logs):
        val_acc = logs['val_acc']

        if(val_acc > self.best_val_acc):
            self.best_val_acc = val_acc
            self.epoch = epoch
            print("Epoch {}: val_acc increased from {:.4f} to {:.4f}, saving model to {}"
                .format(epoch, val_acc, self.best_val_acc, self.path))
            self.model.save(self.path)

    def train_anchor(self, data_train, train_labels, data_validation, validation_labels, data_test, test_labels,
        epochs, batch_size,verbose=0, use_gen=True):
        from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,LambdaCallback
        #path = self.name+"/"+self.name+"-{val_acc:.3f}.hdf5" #+".json" # str path
        #path = self.name+"/"+self.name+"-best.hdf5" #+".json" # str path
        path = self.name+"/"+self.name
        end = "-best.hdf5"
        self.path = str(self.modelpath/path)+end
        #save = ModelCheckpoint(
        #    str(self.modelpath/path), monitor='val_acc', verbose=verbose, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        #print(save.filepath)
        stop = EarlyStopping(monitor="val_loss",mode='min',patience=8,verbose=verbose)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=verbose, min_lr=1e-6, mode='min')
        
        if(use_gen):
            history = self.model.fit_generator(generator(data_train, train_labels, batch_size=batch_size),
            validation_data=(data_validation, validation_labels),
            epochs=epochs, steps_per_epoch=30,callbacks=[LambdaCallback(on_epoch_end=self.checkPoint),stop,reduce_lr])
            #history = self.model.fit(generator(data_train, train_labels, batch_size=32), 
            #validation_data=generator(data_validation, validation_labels, batch_size=200), validation_steps=10,
            #epochs=epochs,steps_per_epoch=10,
            #callbacks=[save,stop,reduce_lr])
        else:
            history = self.model.fit(data_train, train_labels, 
            shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(data_validation, validation_labels),
            callbacks=[LambdaCallback(on_epoch_end=self.checkPoint),stop,reduce_lr])


        # TODO: load best model from ModelCheckpoint
        self.load_model_checkpoint(self.path)

        self.evaluate(data_train=data_train,train_labels=train_labels,
                    data_test=data_test,test_labels=test_labels)
        self.acc = int(self.acc_test*10**4)
        self.rename_model(str(self.modelpath/path))

        # we want to rename the file to something else.
        #path = self.name+"/"+self.name+"-{val_acc:.3f}.hdf5" #+".json" # str path

        # Check if accuracy is good enough.
        self.check_if_persist(str(self.modelpath/path))
        

        # Draw graph of training epochs.
        if(verbose > 0):
            import matplotlib.pyplot as plt
            plt.figure(1)
            
            plt.subplot(131)
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.axvline(x=self.epoch,linestyle=":",color="g")
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['training', 'validation','checkpoint'], loc='lower right')

            plt.subplot(132)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['training', 'validation'], loc='upper left')

            #ROC_curve

            y_pred = self.model.predict(data_test).ravel()
            from sklearn.metrics import roc_curve,auc
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred)
            auc_keras = auc(fpr_keras, tpr_keras)

            plt.subplot(133)
            plt.plot([0, 1], [0, 1], 'k--') # center line
            plt.plot(fpr_keras, tpr_keras,label='Network (area = {:.3f})'.format(auc_keras))
            #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve(checkpint)')
            plt.legend(loc='best')

            # We do not want to store the plot if it has less than 85 % test accuracy
            # get 4 decimal places as integer.
            print("acc",self.acc)
            if(self.acc >= 8500): # only if accuracy is above 85%
                plt.savefig(str(self.modelpath/path)+"-"+str(self.acc)+'.png')
            plt.show()

    def check_if_persist(self,path):
        # Check if accuracy on test set is above 85 % otherwise we delete the model.
        #this is to keep only the best models, at 85% treshold.
        #acc = int(self.acc_test*10**4) # get 4 decimal places as integer.
        if(os.path.isfile(path+"-"+str(self.acc)+".hdf5")):
            if(self.acc < 8500): # if accuracy is less, we simply remove the model
                os.remove(path+"-"+str(self.acc)+".hdf5")
                

    def rename_model(self,path):
        # We want to add accuaracy to the model save-point
        if(os.path.isfile(self.path)):
            #acc = int(self.acc_test*10**4) # get 4 decimal places as integer.
            # check if file allready exists, if so delete previous file
            if(os.path.exists(path+"-"+str(self.acc)+".hdf5")): 
                os.remove(path+"-"+str(self.acc)+".hdf5")
            os.rename(self.path, path+"-"+str(self.acc)+".hdf5")
        else:
            raise ValueError("path is not a file",path)

    def evaluate(self, data_train, train_labels, data_test, test_labels, batch_size=1000):
        score, acc = self.model.evaluate(data_train, train_labels, batch_size=batch_size)
        score_test, self.acc_test = self.model.evaluate(data_test, test_labels, batch_size=batch_size)
        print("loss_train {:.6f}  loss_test {:6f} - acc_train {:.2f}%  acc_test {:.2f}%".format(score, score_test, acc*100, self.acc_test*100))

    
    def predict(self, data): # return an class as a one dimention np.array (c,)
        return self.model.predict_classes(data).flatten()

import numpy as np
def generator(x, y, batch_size):
    while True: # never ending loop
        #Randomly select batch_size number of sentences
        indx = np.random.choice(x.shape[0], batch_size, replace=False)
        yield x[indx], y[indx] # return selected data and corresponding labes

sgd = SGD(lr=0.01)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True)
adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)

from keras.regularizers import l1,l1_l2,l2

def NN_adult(input_dim, output_dim, name="NN-Adult"): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    optimizer = Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True)
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

def NN_adult_1(input_dim, output_dim, name="NN-Adult-1"): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True)
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

def NN_adult_2(input_dim, output_dim, name="NN-Adult-2"): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True)
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

def NN_adult_3(input_dim, output_dim, name="NN-Adult-3"): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    optimizer = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True)
    return Model(model=[
        Dense(128, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.3),
        Dense(128, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l1=0.001,l2=0.001)),
        Dropout(0.3),
        Dense(64, input_dim=input_dim, activation="relu",activity_regularizer=l1(0.001)),
        Dropout(0.2),
        Dense(32, input_dim=input_dim, activation="relu"),
        Dropout(0.1),
        Dense(8, activation="relu"),
        Dense(output_dim,activation="sigmoid") # output layer
        #Activation('sigmoid)
    ],
    optimizer=optimizer, loss="binary_crossentropy",name=name)

def NN_adult_4(input_dim, output_dim, name="NN-Adult-4"): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    optimizer = Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True)
    return Model(model=[
        Dense(128, input_dim=input_dim, activation="relu",activity_regularizer=l2(l=0.001),bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.5),
        Dense(96, input_dim=input_dim, activation="relu",activity_regularizer=l2(l=0.001),bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.4),
        Dense(64, input_dim=input_dim, activation="relu", bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.3),
        Dense(32, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(output_dim,activation="sigmoid") # output layer
        #Activation('sigmoid)
    ],
    optimizer=optimizer, loss="binary_crossentropy",name=name)

def NN_adult_5(input_dim, output_dim, name="NN-Adult-5"): # input -> linear(50) -> relu -> linear(dim*dim) -> softmax
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True)
    return Model(model=[
        Dense(512, input_dim=input_dim, activation="relu",activity_regularizer=l2(l=0.001),bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.6),
        Dense(256, input_dim=input_dim, activation="relu",activity_regularizer=l2(l=0.001),bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.5),
        Dense(128, input_dim=input_dim, activation="relu",activity_regularizer=l2(l=0.001),bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.4),
        Dense(96, input_dim=input_dim, activation="relu",activity_regularizer=l2(l=0.001),bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.3),
        Dense(64, input_dim=input_dim, activation="relu", bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.1),
        Dense(32, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dropout(0.1),
        Dense(16, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dense(8, input_dim=input_dim, activation="relu",bias_regularizer=l1_l2(l2=0.001,l1=0.001)),
        Dense(output_dim,activation="sigmoid") # output layer
        #Activation('sigmoid)
    ],
    optimizer=optimizer, loss="binary_crossentropy",name=name)
