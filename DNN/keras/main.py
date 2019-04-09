import pandas as pd
import numpy as np

import os,sys,inspect
# add parent folder to path ( DNN )
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir) # get path dir
sys.path.insert(0,parentdir) # insert into first spot

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
    
    #pip install shap

    import shap

    explainer = shap.DeepExplainer(model.model,w_train)

    shap_values = explainer.shap_values(w_train[:2])

    print(len(shap_values),shap_values[0].shape)

    print(shap_values)
    print(shap_values[0])

def train_wine_model():
    """ Train wine model for testing other functions """
    from keras.optimizers import SGD,Adam,Adagrad,RMSprop
    from keras import losses

    dataman = Datamanager.Datamanager(dataset="wine")
    
    sgd= SGD(lr=0.01)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # loss functions
    mse = losses.mean_squared_error
    cce = losses.categorical_crossentropy
    model = network.NN_3_20(input_dim=dataman.input_dim,output_dim=dataman.classes,name="wineI",optimizer=adam)
    model.train(dataman,50,20)
    model.store(50)

def test_DeepExplain():
    sgd = SGD(lr=0.01)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # loss functions
    mse = losses.mean_squared_error
    cce = losses.categorical_crossentropy
    # pip install git+https://github.com/marcoancona/DeepExplain.git#egg=deepexplain
    from deepexplain.tensorflow import DeepExplain

    model = network.Model(name="wine") # collect pretrained model from file
    model.compile(loss=mse, optimizer=adam,metrics=['accuracy'])

    dataman = Datamanager.Datamanager(dataset="wine")

    background = dataman.return_background(100) # get 100 examples
    print(background[0][:2],background[1][:2]) # input + target
    print(model.predict(x=background[0][:2]))

    from keras import backend as K
    from keras.models import Sequential, Model
    with DeepExplain(session=K.get_session()) as de: # init DeepExplain context 
        #print("Trying to explain...")
        input_tensors = model.model.layers[0].input
        #print(input_tensors)
        output_layers = model.model.layers[-2].output
        #print(output_layers)
        fModel = Model(inputs =input_tensors, outputs = model.model.layers[-2].output)
        #print(fModel.summary())
        target_tensor = fModel(input_tensors)
        attribution = de.explain('intgrad',target_tensor,input_tensors , background[0][:2])
        print(attribution)

def test_deepLift():
    sgd = SGD(lr=0.01)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # loss functions
    mse = losses.mean_squared_error
    cce = losses.categorical_crossentropy
    # pip install git+https://github.com/marcoancona/DeepExplain.git#egg=deepexplain
    from deepexplain.tensorflow import DeepExplain

    from keras.optimizers import SGD,Adam,Adagrad,RMSprop
    from keras import losses

    model = network.Model(name="wine") # collect pretrained model from file
    model.compile(loss=mse, optimizer=adam,metrics=['accuracy'])

    dataman = Datamanager.Datamanager(dataset="wine")

    background = dataman.return_background(100) # get 100 examples

    import deeplift
    from deeplift.layers import NonLinearMxtsMode
    from deeplift.conversion import kerasapi_conversion as kc

def test_anchors():
    #pip install anchor_exp
    from anchor import utils
    #import utils
    #import anchor_tabular
    from anchor import anchor_tabular

    dataset_folder = "../../Data/"

    # get bunch object, with a dict containing interesting keypoints of the dataset.
    # training_set, validation_set, testing_set, feature_names, categories_per_feature etc.
    dataset = utils.load_dataset("adult",balance=True, dataset_folder=dataset_folder)

    print(dataset.__dict__.keys())

    explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data ,dataset.categorical_names)
    explainer.fit(dataset.train, dataset.labels_train, dataset.validation, dataset.labels_validation)
    print(explainer.encoder)
    print(explainer.disc)
    #print(dataset.__dict__)
    #model = network.Model(name="wine")
    #dataman = Datamanager.Datamanager(dataset="wine")
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, n_jobs=5)
    print(model)
    model.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
    predict_fn = lambda x: model.predict(explainer.encoder.transform(x)) # use the explainer.encoder to transform the data first.
    print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
    print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))

    # Anchor
    idx = 0
    np.random.seed(1)
    print(dataset.test[idx])
    print(dataset.test[idx].reshape(1,-1))

    print("prediction: ", explainer.class_names[predict_fn(dataset.test[idx].reshape(1,-1))[0]]) # predict on the first datapoint    
    exp = explainer.explain_instance(dataset.test[idx], model.predict, threshold=0.95)
    print(exp.names())
    print("Anchor: %s" % (" AND ".join(exp.names())))
    print("Precision: %.2f" % exp.precision())
    print("Coverage: %.2f" % exp.coverage())
    print(exp.features())

    # Check that the ancor holds for other data points.
    all_np = np.all(dataset.test[:, exp.features()] == dataset.test[idx][exp.features()], axis=1)
    print(all_np)
    fit_anchor = np.where((all_np))[0] # select the array of indexes?
    print(fit_anchor,fit_anchor.shape)
    print('Anchor test precision: %.2f' % (np.mean(predict_fn(dataset.test[fit_anchor]) == predict_fn(dataset.test[idx].reshape(1, -1)))))
    print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(dataset.test.shape[0])))

    # Looking at a particular anchor
    print('Partial anchor: %s' % (' AND '.join(exp.names(1))))
    print('Partial precision: %.2f' % exp.precision(1))
    print('Partial coverage: %.2f' % exp.coverage(1))

def test_lore():
    sys.path.append("..")
    from ..DNN.Induction.LORE import lore
    #import DNN.Induction.LORE as lore
    #from ..Induction.LORE import lore
    #from DNN.Induction.LORE import lore

def test_percentile():
    a = np.array([1,2,3,4,5,6,7,8,9,10]) 
    percentile = np.percentile(a,[20,50,70]) # seperate the array, using percentiles.
    # result [2.8 5.5 7.3]. < 2.8, <= 5.5 , 7.3 < 
#train_wine_model()
#test_network()
#test_DeepExplain()

#test_anchors()
