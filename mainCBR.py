import pandas as pd
import numpy as np

from DNN.kera import pre_processing

"""
    1. Data fra datamanger er encoded til tall-labels. 
       Må decode test-settet tilbake til kategori.
    2. Må deretter reversere diskretiseringen for de to kolonnene med int-features.
    3. Legg inn i case-basen. 
"""

def complete():
    from DNN.Induction.Anchor import anchor_tabular, utils

    dataman = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
    dataset = dataman.ret

    # IMPORT THE NETWORK
    # Fit explainer to the same dataset it was trained on
    explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data_train, dataset.categorical_names)
    # Explainer.encoder.transform return sparse matrix, instead of dense np.array
    explainer.fit(dataset.data_train, dataset.train_labels, dataset.data_validation, dataset.validation_labels)

    # LOAD MODEL
    from DNN.kera import network
    bb = network.BlackBox(name="NN-adult-5",c_path="NN-Adult-5/NN-Adult-5-8531.hdf5")
    bb.evaluate(data_train=explainer.encoder.transform(dataset.data_train).toarray(),train_labels=dataset.train_labels, data_test=explainer.encoder.transform(dataset.data_test).toarray(),test_labels=dataset.test_labels)
    
    predict_fn = lambda x: bb.predict(explainer.encoder.transform(x))

    idx = 0
    print(len(dataset.data_test))
    instance = dataset.data_test[idx].reshape(1,-1)
    prediction = predict_fn(instance)[0]


    from DNN import explanation
    from DNN import knowledge_base

    exp = explainer.explain_instance(instance, bb.predict, threshold=0.98,verbose=True)
    print(exp.exp_map.keys()) 
    
    exp_1 = explanation.Explanation(**exp.exp_map)
    print(exp_1)
    print()
    print(exp_1.get_explanation(dataset.feature_names,dataset.categorical_names))

# Process data before adding to case-base
def post_process(explanationForNCases, verbose):
    dataman = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
    dataset = dataman.ret


    decoded = []
    # Convert encoded labels back to string feature, e.g. [3, 0,..., 8] --> ['Married', 'Sales',..., 'White']
    for row in dataset.data_test:
        translated_row = dataman.translate(row=row)
        decoded.append(translated_row)
    decoded = np.array(decoded)

    # Convert to dataframe (easier to work with)
    names = ["age", "workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "country"]
    df = pd.DataFrame(decoded, columns=np.array(names))
    if verbose: print(df.head())

    # Find de-discretisized int columns in original dataset (before test/val split)
    df_int_columns = dataset.data_test_full.iloc[dataset.test_idx, :]
    if verbose: print(df_int_columns.head())

    # Reverse discretisized int columns  (go from interval to integer)
    df.age = df_int_columns['age'].tolist()
    df.hours_per_week = df_int_columns['hours per week'].tolist()
    if verbose: print(df.head(10))


    # ------------------------ EXPLANATION PART -----------------------------
    from DNN.Induction.Anchor import anchor_tabular, utils
    from DNN.kera import network
    
    # IMPORT THE NETWORK
    # Fit explainer to the dataset it was trained on
    explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data_train, dataset.categorical_names)
    explainer.fit(dataset.data_train, dataset.train_labels, dataset.data_validation, dataset.validation_labels)

    # LOAD MODEL
    bb = network.BlackBox(name="NN-adult-5",c_path="NN-Adult-5/NN-Adult-5-8531.hdf5")
    # Explainer.encoder.transform return sparse matrix, instead of dense np.array
    bb.evaluate(data_train=explainer.encoder.transform(dataset.data_train).toarray(),train_labels=dataset.train_labels, data_test=explainer.encoder.transform(dataset.data_test).toarray(),test_labels=dataset.test_labels)
    predict_fn = lambda x: bb.predict(explainer.encoder.transform(x))

    from DNN import knowledge_base
    from DNN import explanation

    explanations = []
    def explain_instances():
        explanations_as_string = []
        # NOTE: Change slice in for-loop to decide how many instances are going to be explained
        for instance in dataset.data_test[:explanationForNCases]:
            exp = explainer.explain_instance(instance, bb.predict, threshold=0.98,verbose=False)
            exp_1 = explanation.Explanation(**exp.exp_map)
            explanations.append(exp_1)
            # print('exp_1:',exp_1)
            interpreted_expl = exp_1.get_explanation(dataset.feature_names,dataset.categorical_names)
            explanations_as_string.append(interpreted_expl)
        return explanations_as_string
            
    explained = explain_instances()
    # print(explained)
    explained = explained + ['__unknown__' for i in range(len(dataset.data_test) - len(explained))]

    df['explanation'] = np.array(explained)
    if verbose: print(df.head(10))

    return df, explanations



# Add cases to case-base
def populate_casebase(n_cases=2):
    from itertools import islice
    import CBR.src.CBRInterface as CBRInterface
    
    CBR = CBRInterface.RESTApi()

    # Process data before adding to case-base
    # Explanations as Explanation-objects
    df, explanations = post_process(explanationForNCases=n_cases, verbose=False) 
    # df, explanations = post_process(verbose=False)          

    
    print('explanaaations:', explanations)

    # Formatted JSON which is passed in as params to REST
    instanceJson = lambda row: {
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
            "Explanation": row['explanation']
        }]
    }
    
    print('These instances were added:')
    for index, row in islice(df.iterrows(), n_cases):
        # Add instance to CB
        # result = CBR.addInstancesJSON(casebaseID='cb0',conceptID='Person',cases=instanceJson(row))
        # print(result)

        # Add to KB
        from DNN.knowledge_base import KnowledgeBase
        print(explanations[index])
        # kb1 = KnowledgeBase('kb1')
        # kb1.add_knowledge(explanations[index])
        # print(index)
        # pass


populate_casebase(n_cases=2)


# TODO: Legge inn data fra valideringssettet i case-basen og bruke test-settet kun til å querye nye cases



# Grammarly
# ------
# Hvis vi skal ha knowledge-base må vi fortsatt ha pekere..?
# ------
# Vil gjerne ha kommentarer på om det er noen subsections som bør komme før/etter andre. 
# Spesielt hvor tidlig final architecture-seksjonen bør komme i kapittel 4.