
def test_lore():
    """ Quick test of lore package """
    #import DNN.Induction.LORE as LORE
    from DNN.Induction.LORE import lore
    from DNN.Induction.LORE import evaluation
    from DNN.Induction.LORE import prepare_dataset
    from DNN.Induction.LORE import util

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    import warnings
    warnings.filterwarnings("ignore") # hide SettingWithCopyWarning
    
    dataset = prepare_dataset.prepare_adult_dataset("adult.csv","Data/adult/")

    X,y = dataset['X'],dataset['y']
    
    #Split matrices into random random train and test susets. random_state is seed.
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

    blackbox = RandomForestClassifier(n_estimators=20)
    blackbox.fit(x_train,y_train)
    print("Done training")
    X2E = x_test
    idx_record2explain = 1 #
    explanation, infos = lore.explain(idx_record2explain=idx_record2explain, X2E=X2E, dataset=dataset, blackbox=blackbox,
                                    discrete_use_probabilities=True,
                                    continuous_function_estimation=True,
                                    returns_infos=True)
    print("Done explaining")
    print(explanation, infos)

    x = util.build_df2explain(blackbox, X2E[idx_record2explain].reshape(1,-1), dataset).to_dict('records')[0]

    print("x = %s" % x)
    print("r = %s --> %s" % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)

    print('Evaliation')
    bb_outcome = infos['bb_outcome']
    cc_outcome = infos['cc_outcome']
    y_pred_bb = infos['y_pred_bb']
    y_pred_cc = infos['y_pred_cc']
    dfZ = infos['dfZ']
    dt = infos['dt']
    tree_path = infos['tree_path']
    leaf_nodes = infos['leaf_nodes']
    diff_outcome = infos['diff_outcome']

    print(tree_path)

    print(evaluation.evaluate_explanation(x, blackbox, dfZ, dt, tree_path, leaf_nodes, bb_outcome, cc_outcome,
                            y_pred_bb, y_pred_cc, diff_outcome, dataset, explanation[1]))

#   Step 1: 
#       Preprocess dataset: Discretazise, and relable.
#   Step 2:
#       Fit the dataset to the explainer. 
#   Step 3:
#       Fit BlackBox to the dataset.
#   Step 4:
#       Use explainer to explain decision of black-box.
#   Step 5:
#       Store explanation along with input/output in CBR system.
#   Step 6:
#       Utilise explanation/prediction on new problems.

def test_anchors():
    import anchor

    import numpy as np
    # ! Old imports
    #pip install anchor_exp
    from anchor import utils, anchor_tabular
    #from DNN.Induction import Anchor
    #from DNN.Induction.Anchor import anchor_tabular, utils
    dataset_folder = "Data/"

    # get bunch object, with a dict containing interesting keypoints of the dataset.
    # training_set, validation_set, testing_set, feature_names, categories_per_feature etc.
    dataset = utils.load_dataset("adult",balance=True, dataset_folder=dataset_folder)

    print(dataset.__dict__.keys())
    print(dataset.categorical_names)
    explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data, dataset.categorical_names)
    explainer.fit(dataset.train, dataset.labels_train, dataset.validation, dataset.labels_validation,
    discretizer='quartile')
    exit()
    print(explainer.encoder)
    print(explainer.encoder.transform)
    #print(dataset.__dict__)
    #model = network.Model(name="wine")
    #dataman = Datamanager.Datamanager(dataset="wine")
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, n_jobs=5)
    print(model)
    print(explainer.encoder.transform(dataset.train))#, dataset.labels_train
    model.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
    predict_fn = lambda x: model.predict(explainer.encoder.transform(x)) # use the explainer.encoder to transform the data first.
    print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
    print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))
    
    # Anchor
    idx = 0
    np.random.seed(1)
    print("Instance to explain",dataset.test[idx].reshape(1,-1))
    prediction_class = predict_fn(dataset.test[idx].reshape(1,-1))[0] # select first index of prediction matrix.
    print("prediction: ", explainer.class_names[prediction_class])

    exp = explainer.explain_instance(dataset.test[idx], model.predict, threshold=0.95)
    print(explainer.explain_instance)
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
    print('Partial anchor: %s' % (' AND '.join(exp.names(1)))) # get part of the anchor.
    print('Partial precision: %.2f' % exp.precision(1))
    print('Partial coverage: %.2f' % exp.coverage(1))

    print(exp.features())
    print(dataset.test[:, exp.features(1)] == dataset.test[idx][exp.features(1)])
    fit_partial = np.where(np.all(dataset.test[:, exp.features(1)] == dataset.test[idx][exp.features(1)], axis=1))[0]
    print('Partial anchor test precision: %.2f' % (np.mean(predict_fn(dataset.test[fit_partial]) == predict_fn(dataset.test[idx].reshape(1, -1)))))
    print('Partial anchor test coverage: %.2f' % (fit_partial.shape[0] / float(dataset.test.shape[0])))

def test_anchors_nn():
    import numpy as np
    # ? copy from repository
    from DNN.Induction.Anchor import anchor_tabular, utils

    from DNN.keras import pre_processing
    dataset_folder = "Data/"
    dataset_2 = utils.load_dataset("adult",balance=True, dataset_folder=dataset_folder)
    datamanager = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
    dataset = datamanager.ret

    # TODO: Finn ulikhet i dataset som foresaker feil..
    #print(dataset_2.class_names)
    #print(dataset.class_names)

    #print(dataset_2.labels)
    #print(dataset.labels)

    #print(dataset_2.categorical_features)
    #print(dataset.categorical_features)

    #print(dataset_2.categorical_names[11])
    #print(dataset.categorical_names[11])

    #print(dataset_2.ordinal_features)
    #print(dataset.ordinal_features)

    #print(dataset_2.feature_names)
    #print(dataset.feature_names)

    #print(dataset_2.train,type(dataset_2.train),type(dataset_2.train[0]),type(dataset_2.train[0][0]))
    #print(dataset.data_train,type(dataset.data_train),type(dataset.data_train[0]),type(dataset.data_train[0][0]))

    #print(dataset_2.labels_train)
    #print(dataset.train_labels)

    
    #dataman = preprocessing.datamanager()

    # Fit the explainer to the dataset. 
    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data_train, dataset.categorical_names)
        
    explainer.fit(dataset.data_train, dataset.train_labels, 
                dataset.data_validation, dataset.validation_labels)

    explainer_2 = anchor_tabular.AnchorTabularExplainer(
        dataset_2.class_names, dataset_2.feature_names,
        dataset_2.data, dataset_2.categorical_names
    )

    explainer_2.fit(dataset_2.train, dataset_2.labels_train, dataset_2.validation, dataset_2.labels_validation)

    #print(explainer.encoder.transform)
    #print(explainer.disc)
    #print(dataset.__dict__)
    #model = network.Model(name="wine")
    #dataman = Datamanager.Datamanager(dataset="wine")
    
    #print(dataset.data_train[0])
    #print(explainer.encoder.transform(dataset.data_train)[0].shape)
    #print(explainer.encoder.transform(dataset.data_train)[0].toarray())
    #print(explainer.encoder.transformers[0])
    import sklearn
    if(True):
        from DNN.keras import network
        #print(dataset.categorical_names, dataset.categorical_names.keys())
        n_values = sum([len(dataset.categorical_names[i]) for i in dataset.categorical_names.keys()])
        nn = network.NN_adult_2(n_values,1)
        nn.train_anchor(explainer.encoder.transform(dataset.data_train), dataset.train_labels,
                explainer.encoder.transform(dataset.data_validation), dataset.validation_labels,
                epochs=10, batch_size=64)
        
        model = nn
        # ? Load pretrained model..
        #model = network.Model(name="adults")
        # use the explainer.encoder to transform the data first.
        predict_fn = lambda x: model.predict(explainer.encoder.transform(x)) 

        print(dataset.data_train.shape, explainer.encoder.transform(dataset.data_train).shape)

        print('Train', sklearn.metrics.accuracy_score(dataset.train_labels, predict_fn(dataset.data_train)))
        print('Test', sklearn.metrics.accuracy_score(dataset.test_labels, predict_fn(dataset.data_test)))

    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, n_jobs=5)
        print(model)
        #print(explainer.encoder.transform(dataset.data_train))#, dataset.labels_train
        model.fit(explainer.encoder.transform(dataset.data_train), dataset.train_labels)
        predict_fn = lambda x: model.predict(explainer.encoder.transform(x)) # use the explainer.encoder to transform the data first.
        print(dataset.data_train.shape, explainer.encoder.transform(dataset.data_train).shape)

        print('Train', sklearn.metrics.accuracy_score(dataset.train_labels, predict_fn(dataset.data_train)))
        print('Test', sklearn.metrics.accuracy_score(dataset.test_labels, predict_fn(dataset.data_test)))

    idx = 1
    np.random.seed(1)
    print("predicting", dataset.data_test[idx].reshape(1,-1)[0])
    prediction = predict_fn(dataset.data_test[idx].reshape(1,-1))[0]
    print("prediction:", prediction,"=",explainer.class_names[prediction])
    #print("prediction: ", explainer.class_names[predict_fn(dataset.data_test[idx].reshape(1,-1))[0]]) # predict on the first datapoint    
    exp = explainer.explain_instance(dataset.data_test[idx], model.predict, threshold=0.98,verbose=True)
    #print(exp.names())
    print("Anchor: %s" % (" AND ".join(exp.names())))
    print("Precision: %.2f" % exp.precision())
    print("Coverage: %.2f" % exp.coverage())
    print(exp.features())

    exit()
    # TODO: put explainer encoder in pre_processor 

    model.fit(explainer.encoder.transform(dataset.data_train), dataset.train_labels)
    predict_fn = lambda x: model.predict(explainer.encoder.transform(x)) # use the explainer.encoder to transform the data first.
    print('Train', sklearn.metrics.accuracy_score(dataset.train_labels, predict_fn(dataset.data_train)))
    print('Test', sklearn.metrics.accuracy_score(dataset.test_labels, predict_fn(dataset.data_test)))
    # Anchor
    idx = 0
    np.random.seed(1)
    print(dataset.test_labels[idx])
    print(dataset.test_labels[idx].reshape(1,-1))

    print("prediction: ", explainer.class_names[predict_fn(dataset.data_test[idx].reshape(1,-1))[0]]) # predict on the first datapoint    
    exp = explainer.explain_instance(dataset.data_test[idx], model.predict, threshold=0.95)
    print(exp.names())
    print("Anchor: %s" % (" AND ".join(exp.names())))
    print("Precision: %.2f" % exp.precision())
    print("Coverage: %.2f" % exp.coverage())
    print(exp.features())

    # TODO: list of catagories -> encoding -> one_hot_encoding.
    exit()
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


def test_anchor_nn_data():
    import numpy as np
    # ? copy from repository
    from DNN.Induction.Anchor import anchor_tabular, utils

    from DNN.keras import pre_processing
    dataset_folder = "Data/"
    dataset = utils.load_dataset("adult",balance=True, dataset_folder=dataset_folder)
    print(dataset.__dict__.keys())
    datamanager = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
    #dataset = datamanager.ret

    #print(dataset_2.categorical_names[11])
    print(dataset.categorical_names[11])

    #print(dataset_2.ordinal_features)
    print(dataset.ordinal_features)

    #print(dataset_2.feature_names)
    print(dataset.feature_names)

    #print(dataset_2.train,type(dataset_2.train),type(dataset_2.train[0]),type(dataset_2.train[0][0]))
    #print(dataset.data_train,type(dataset.data_train),type(dataset.data_train[0]),type(dataset.data_train[0][0]))

    #print(dataset_2.labels_train)
    #print(dataset.train_labels)

    
    #dataman = preprocessing.datamanager()

    # Fit the explainer to the dataset. 
    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.train, dataset.categorical_names
    )

    explainer.fit(dataset.train, dataset.labels_train, dataset.validation, dataset.labels_validation)

    print(explainer.encoder.transform)
    print(explainer.disc)
    #print(dataset.__dict__)
    #model = network.Model(name="wine")
    #dataman = Datamanager.Datamanager(dataset="wine")
    
    #print(dataset.data_train[0])
    #print(explainer.encoder.transform(dataset.data_train)[0].shape)
    #print(explainer.encoder.transform(dataset.data_train)[0].toarray())
    #print(explainer.encoder.transformers[0])
    import sklearn
    if(True): # IF network.
        from DNN.keras import network

        nn = network.NN_adult_2(123,1)
        #dataset_2.train, dataset_2.labels_train, dataset_2.validation, dataset_2.labels_validation
        nn.train_anchor(explainer.encoder.transform(dataset.train), dataset.labels_train,
                explainer.encoder.transform(dataset.validation), dataset.labels_validation,
                epochs=1, batch_size=100)
        
        model = nn
        # ? Load pretrained model..
        #model = network.Model(name="adults")
        predict_fn = lambda i: model.predict(explainer.encoder.transform(i)) # use the explainer.encoder to transform the data first.
        print(dataset.train.shape, explainer.encoder.transform(dataset.train).shape)
        print(predict_fn(dataset.train).shape, type(predict_fn(dataset.train)),
        predict_fn(dataset.train))

        print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
        print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, n_jobs=5)
        print(model)
        #print(explainer.encoder.transform(dataset.data_train))#, dataset.labels_train
        model.fit(explainer.encoder.transform(dataset.data_train), dataset.train_labels)
        predict_fn = lambda x: model.predict(explainer.encoder.transform(x)) # use the explainer.encoder to transform the data first.
        print(dataset.data_train.shape, explainer.encoder.transform(dataset.data_train).shape)
        print(predict_fn(dataset.data_train).shape, type(predict_fn(dataset.data_train)),predict_fn(dataset.data_train))
        print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
        print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))

    idx = 0
    np.random.seed(1)
    print(predict_fn(dataset.test[idx].reshape(1,-1))[0])
    prediction = predict_fn(dataset.test[idx].reshape(1,-1))[0]
    print(explainer.class_names)
    print("prediction:", explainer.class_names[prediction])

    #print("prediction: ", explainer.class_names[predict_fn(dataset.data_test[idx].reshape(1,-1))[0]]) # predict on the first datapoint    
    exp = explainer.explain_instance(dataset.test[idx], model.predict, threshold=0.95)
    print(exp.names())
    print("Anchor: %s" % (" AND ".join(exp.names())))
    print("Precision: %.2f" % exp.precision())
    print("Coverage: %.2f" % exp.coverage())
    print(exp.features())

    exit()
    # TODO: put explainer encoder in pre_processor 

    model.fit(explainer.encoder.transform(dataset.data_train), dataset.train_labels)
    predict_fn = lambda x: model.predict(explainer.encoder.transform(x)) # use the explainer.encoder to transform the data first.
    print('Train', sklearn.metrics.accuracy_score(dataset.train_labels, predict_fn(dataset.data_train)))
    print('Test', sklearn.metrics.accuracy_score(dataset.test_labels, predict_fn(dataset.data_test)))
    # Anchor
    idx = 0
    np.random.seed(1)
    print(dataset.test_labels[idx])
    print(dataset.test_labels[idx].reshape(1,-1))

    print("prediction: ", explainer.class_names[predict_fn(dataset.data_test[idx].reshape(1,-1))[0]]) # predict on the first datapoint    
    exp = explainer.explain_instance(dataset.data_test[idx], model.predict, threshold=0.95)
    print(exp.names())
    print("Anchor: %s" % (" AND ".join(exp.names())))
    print("Precision: %.2f" % exp.precision())
    print("Coverage: %.2f" % exp.coverage())
    print(exp.features())

    

    # TODO: list of catagories -> encoding -> one_hot_encoding.
    exit()
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

def test_integrated_gradients():
    # https://github.com/hiranumn/IntegratedGradients/blob/master/IntegratedGradients.py

    import numpy as np
    np.random.seed(1) # set seed

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Create a simple model to test out.
    from keras.optimizers import SGD,Adam,Adagrad,RMSprop
    from keras import losses

    sgd = SGD(lr=0.01)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
    # loss functions
    mse = losses.mean_squared_error
    cce = losses.categorical_crossentropy
    from DNN.keras import network

    bb = network.Model(name="wine") # collect pretrained model from file
    bb.compile(loss=mse, optimizer=adam,metrics=['accuracy'])
    
    import datamanager
    dataman = datamanager.Datamanager(dataset="wine")
    
    #Implement IntegradetGrad
    from DNN.Induction.IntGrad import IntegratedGradients  
    t_inputs, t_targets = dataman.return_keras()

    #total = bb.predict(X[0:1, :])[0,0]-model.predict(np.zeros((1,4)))[0,0]
    
    ig = IntegratedGradients.integrated_gradients(bb.model, verbose=0)


    print("Explaning:", t_inputs[0])
    attribution = ig.explain(t_inputs[0], num_steps=100)
    print("sum Attribution:",sum(attribution))

    attribution = ig.explain(t_inputs[0], num_steps=100000)
    print("sum Attribution:",sum(attribution))
    #print("TEST")

    #explanation = ig.explain(X[0], num_steps=100000)
    #print(total, explanation)

def test_autoencoder():
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)
    print(enc.categories_)

    n_values = [4, 9, 16, 7, 15, 6, 5, 2, 3, 3, 3, 42] # values per category
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def train_network():
    import numpy as np
    np.random.seed(1) 
    import tensorflow as tf
    tf.set_random_seed(1)

    import sklearn
    from DNN.keras import pre_processing
    from DNN.Induction.Anchor import anchor_tabular, utils
    
    
    datamanager = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
    dataset = datamanager.ret

    # Fit the explainer to the dataset. 
    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data_train, dataset.categorical_names)
        
    explainer.fit(dataset.data_train, dataset.train_labels, 
                dataset.data_validation, dataset.validation_labels)

    from DNN.keras import network
    
    #keras.random.seed(1)
        #print(dataset.categorical_names, dataset.categorical_names.keys())
    n_values = sum([len(dataset.categorical_names[i]) for i in dataset.categorical_names.keys()])
    model = network.NN_adult_3(n_values,1)
    model.train_anchor(explainer.encoder.transform(dataset.data_train).toarray(), dataset.train_labels,
            explainer.encoder.transform(dataset.data_validation).toarray(), dataset.validation_labels,
            epochs=100, batch_size=64)
    predict_fn = lambda x: model.predict(explainer.encoder.transform(x)) 
    model.evaluate(data_train=explainer.encoder.transform(dataset.data_train),train_labels=dataset.train_labels,
                    data_test=explainer.encoder.transform(dataset.data_test),test_labels=dataset.test_labels)
    
    #print('Train', sklearn.metrics.accuracy_score(dataset.train_labels, predict_fn(dataset.data_train)))
    #print('Test', sklearn.metrics.accuracy_score(dataset.test_labels, predict_fn(dataset.data_test)))
#test_lore()
#test_anchors()
#test_anchors_nn()
#test_anchor_nn_data() # Ikke datasette sin feil
#test_integrated_gradients()

#test_autoencoder()

train_network()