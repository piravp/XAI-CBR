import network
import Datamanager
import pandas as pd

import os,sys,inspect
# add parent folder to path ( DNN )
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
def test_network():
    from keras.optimizers import SGD,Adam,Adagrad,RMSprop
    from keras import losses
    dataman = Datamanager.Datamanager(classes=3,dataset="wine")
    sgd = SGD(lr=0.01)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # loss functions
    mse = losses.mean_squared_error
    cce = losses.categorical_crossentropy

    model = network.Model(name="wine")

    model.compile(loss=mse, optimizer=adam,metrics=['accuracy'])
    """
    from keras.datasets import imdb
    from keras.preprocessing import sequence
    maxlen = 80
    max_features = 20000
    (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print(x_train[:100])
    print(type(x_train[:100]),type(x_train[0]))
    """

def test_shapley_deepExplainer():
    from keras.optimizers import SGD,Adam,Adagrad,RMSprop
    from keras import losses
    dataman = Datamanager.Datamanager(classes=3,dataset="wine")
    sgd= SGD(lr=0.01)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # loss functions
    mse = losses.mean_squared_error
    cce = losses.categorical_crossentropy

    model = network.Model(name="wine")

    model.compile(loss=mse, optimizer=adam,metrics=['accuracy'])

    background = dataman.return_background(100)
    w_train, w_test = background[0],background[1]

    import shap

    explainer = shap.DeepExplainer(model.model,w_train)

    shap_values = explainer.shap_values(w_train[:2])

    print(len(shap_values),shap_values[0].shape)

    print(shap_values)
    print(shap_values[0])


test_network()