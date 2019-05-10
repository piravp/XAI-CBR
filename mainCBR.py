import pandas as pd
import numpy as np

from DNN.keras import pre_processing

"""
    1. Data fra datamanger er encoded til tall-labels. 
       Må decode test-settet tilbake til kategori.
    2. Må deretter reversere diskretiseringen for de to kolonnene med int-features.
    3. Legg inn i case-basen. 
"""

# Process data before adding to case-base
def post_process(verbose=True):
    
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

    # Reverse discretisized int columns
    df.age = df_int_columns['age'].tolist()
    df.hours_per_week = df_int_columns['hours per week'].tolist()
    if verbose: print(df.head(10))

    return df


# Add cases to case-base
def populate_casebase(n_cases=2):
    from itertools import islice
    import CBR.src.interface as CBRInterface
    
    CBR = CBRInterface.RESTApi()

    # Process data before adding to case-base
    df = post_process(verbose=False)
    
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
            "Country": row['country']
        }]
    }
    
    print('These instances were added:')
    for _, row in islice(df.iterrows(), n_cases):
        print(CBR.addInstancesJSON(casebaseID='cb0',conceptID='Person',cases=instanceJson(row)))


populate_casebase(n_cases=2)
