""" 
    This is the main file used for the experimentation.
    By seperating the experiments and seeding the randomization at each step, it will be reproduceable.

    Its important that the myCBR rest projet is running.
    # COMMAND TO START PROGRAM


"""

import numpy as np

# set seed from numpy to 1.
from DNN.kera import network # Import black-box (ANN)
from DNN.kera import pre_processing # Import dataset preprocessing
from DNN.Induction.Anchor import anchor_tabular, utils # explanatory framework

def exp_case_base_size():
    """ Test initializing casebase of different sizes and querying new problems. """
    np.random.seed(1) # init seed

    dataman = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
    dataset = dataman.ret

    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data_train, dataset.categorical_names)
        
    # ! Explainer.encoder.transform return sparse matrix, instead of dense np.array
    explainer.fit(dataset.data_train, dataset.train_labels, 
                dataset.data_validation, dataset.validation_labels) # Fit the labels to the explainer.
    print(network)

    bb = network.BlackBox(name="NN-adult-5",c_path="NN-Adult-5/NN-Adult-5-8531.hdf5")

    bb.evaluate(data_train=explainer.encoder.transform(dataset.data_train).toarray(),train_labels=dataset.train_labels,
                    data_test=explainer.encoder.transform(dataset.data_test).toarray(),test_labels=dataset.test_labels)

    exit()
    # Check if REST api is running with project.
    CBR = CBRInterface.RESTApi() # load CBR restAPI class, for easy access.


exp_case_base_size()
