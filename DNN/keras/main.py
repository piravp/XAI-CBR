import network
import Datamanager
import pandas as pd

import os,sys,inspect
# add parent folder to path ( DNN )
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from keras.optimizers import SGD,Adam,Adagrad,RMSprop
def test_network():
    dataman = Datamanager.Datamanager(classes=3,dataset="wine")
    sgd = SGD(lr=0.01)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model = network.NN_3_20(name="wine", input_dim=13, classes=3,optimizer=adam) 



    model.train(dataman,50,20)


test_network()