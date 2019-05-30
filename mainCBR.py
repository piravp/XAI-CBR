import pandas as pd
import numpy as np

# Turn off warnings
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from DNN.kera import pre_processing

"""
    1. Data fra datamanger er encoded til tall-labels. 
       Må decode test-settet tilbake til kategori.
    2. Må deretter reversere diskretiseringen for de to kolonnene med int-features.
    3. Legg inn i case-basen. 
"""

# Process data before adding to case-base
def post_process(explanationForNCases, verbose):
    dataman = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
    dataset = dataman.ret


    cat_names = sorted(dataset.categorical_names.keys())
    n_values = [len(dataset.categorical_names[i]) for i in cat_names]


    decoded = []
    # Convert encoded labels back to string feature, e.g. [3, 0,..., 8] --> ['Married', 'Sales',..., 'White']
    for row in dataset.data_test:
        translated_row = dataman.translate(row=row)
        decoded.append(translated_row)
    decoded = np.array(decoded)

    # Convert to dataframe (easier to work with)
    names = ["age", "workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "country"]
    df = pd.DataFrame(decoded, columns=np.array(names))
    if verbose > 1: print(df.head())

    # Find de-discretisized int columns in original dataset (before test/val split)
    df_int_columns = dataset.data_test_full.iloc[dataset.test_idx, :]
    if verbose > 1: print(df_int_columns.head())

    # Reverse discretisized int columns  (go from interval to integer)
    df.age = df_int_columns['age'].tolist()
    df.hours_per_week = df_int_columns['hours per week'].tolist()
    if verbose > 1: print(df.head(10))

    # ----------------------------------------------------------------------------------------------------- #
    #                                       D E E P  E X P L A I N                                          #
    #                                   Generate integrated gradients                                       #
    # ----------------------------------------------------------------------------------------------------- #
    from DNN.kera import network
    from DNN.Induction.Anchor import anchor_tabular

    # LOAD TRAINED MODEL
    bb = network.BlackBox(name="NN-adult-5",c_path="NN-Adult-5/NN-Adult-5-8531.hdf5")

    # Fit explainer to the dataset it was trained on
    explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data_train, dataset.categorical_names)
    explainer.fit(dataset.data_train, dataset.train_labels, dataset.data_validation, dataset.validation_labels)

    from deepexplain.tensorflow import DeepExplain
    from keras import backend as K
    from keras.models import Model

    def generate_integrated_gradient():
        attribution_weights_full = []
        with DeepExplain(session=K.get_session()) as de:
            input_tensors = bb.model.inputs
            output_layer = bb.model.outputs
            fModel = Model(inputs=input_tensors, outputs=output_layer)
            target_tensor = fModel(input_tensors)

            attribution_weights_instance = []
            for instance in dataset.data_test[:explanationForNCases]:
                instance = instance.reshape(1,-1)
                attribution = de.explain('intgrad',target_tensor, input_tensors, explainer.encoder.transform(instance).toarray())
                # print(explainer.encoder.transform(instance).toarray().flatten(),"\nattributions:\n", attribution[0][0],"\n",sum(attribution[0][0]))

                # Compress attribution vector (71 elements, based on one-hot-vector) to only needing 12 elements
                one_hot_vector = attribution[0][0]
                start = 0
                for n in n_values:
                    compressed = sum(one_hot_vector[start:start+n])
                    attribution_weights_instance.append(np.around(compressed, decimals=4))
                    start += n                                                          # increase start slice

                attribution_weights_full.append(str(attribution_weights_instance))      # need to be converted to string to save in case-base
                attribution_weights_instance = []                                       # reset
        return attribution_weights_full
        
    integrated_gradients = generate_integrated_gradient()

    integrated_gradients = integrated_gradients + ['__unknown__' for i in range(len(dataset.data_test) - len(integrated_gradients))]        # Fill default values
    df['weight'] = np.array(integrated_gradients)


    # ----------------------------------------------------------------------------------------------------- #
    #                              G E N E R A T E  E X P L A N A T I O N S                                 #
    #              Generate Explanation objects and populate df with prediction from network                #
    # ----------------------------------------------------------------------------------------------------- #
    from DNN import explanation

    # Explainer.encoder.transform return sparse matrix, instead of dense np.array
    bb.evaluate(data_train=explainer.encoder.transform(dataset.data_train).toarray(),train_labels=dataset.train_labels, data_test=explainer.encoder.transform(dataset.data_test).toarray(),test_labels=dataset.test_labels)
    predict_fn = lambda x: bb.predict(explainer.encoder.transform(x))

    explanations = []
    predictions = []
    def get_prediction():
        # explanations_as_string = []
        # NOTE: Change slice in for-loop to decide how many instances are going to be explained
        for instance in dataset.data_test[:explanationForNCases]:
            exp = explainer.explain_instance(instance, bb.predict, threshold=0.98,verbose=False)
            exp_1 = explanation.Explanation(**exp.exp_map)
            explanations.append(exp_1)
            predictions.append(exp_1.exp_map['prediction'])                                                   # Extract prediction
            # interpreted_expl = exp_1.get_explanation(dataset.feature_names,dataset.categorical_names)       # Explanation as string
            # explanations_as_string.append(interpreted_expl)                                                   
        # return explanations_as_string

    get_prediction()

    predictions = predictions + ['__unknown__' for i in range(len(dataset.data_test) - len(predictions))] 
    df['prediction'] = np.array(predictions)

    # explained = get_prediction()
    # explained = explained + ['__unknown__' for i in range(len(dataset.data_test) - len(explained))]         # Fill default values
    # df['explanation'] = np.array(explained)
    if verbose > 1: print(df.head(10))
    # print(df.head(10))


    return df, explanations


# Add cases to case-base
def populate_casebase(n_cases=2, verbose=1):
    from itertools import islice
    import CBR.src.CBRInterface as CBRInterface
    
    CBR = CBRInterface.RESTApi()

    # Process data before adding to case-base
    # Explanations as Explanation-objects
    df, explanations = post_process(explanationForNCases=n_cases, verbose=verbose) 
    # df, explanations = post_process(verbose=False)          


    # Formatted JSON which is passed in as params to REST
    instanceJson = lambda row, explanation: {
        "cases":[{
            "Age": row['age'],
            "Workclass": row['workclass'],
            "Education": row['education'],
            "MaritalStatus": row['marital_status'],
            "Occupation": row['occupation'],
            "Relationship": row['relationship'],
            "Race": row['race'],
            "Sex": row['sex'],
            "CapitalGain": row['capital_gain'],
            "CapitalLoss": row['capital_loss'],
            "HoursPerWeek": row['hours_per_week'],
            "Country": row['country'],
            "Weight": row['weight'],
            "Prediction": row['prediction'],
            "Explanation": explanation
        }]
    }
    
    # Only add as many cases from df which is specified by user
    for index, row in islice(df.iterrows(), n_cases):
        # Add to KB
        from DNN.knowledge_base import KnowledgeBase
        kb1 = KnowledgeBase('kb1')
        pointer_to_explanation = kb1.add_knowledge(explanations[index])         # Unique id pointing to knowledgeb-base to find explanation
        case_as_json = instanceJson(row, pointer_to_explanation)

        # Add instance to CB
        result = CBR.addInstancesJSON(casebaseID='cb0',conceptID='Person',cases=case_as_json)
        if verbose > 0: print(); print('This instance was added:')
        if verbose > 0: print(result, '\n')




populate_casebase(n_cases=2, verbose=1)     # verbose: <0, 1, 2>


# TODO: Legge inn data fra valideringssettet i case-basen og bruke test-settet kun til å querye nye cases



# Grammarly
# ------
# Hvis vi skal ha knowledge-base må vi fortsatt ha pekere..?
# ------
# Vil gjerne ha kommentarer på om det er noen subsections som bør komme før/etter andre. 
# Spesielt hvor tidlig final architecture-seksjonen bør komme i kapittel 4.