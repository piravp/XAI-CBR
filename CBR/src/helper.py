import json

def pprint(raw_json):
    """
        Pretty print JSON
    """
    return json.dumps(raw_json, indent=4, sort_keys=True)

caseAsJson = lambda row: {
    "cases":[{
        "Age": row['Age'],
        "Workclass": row['Workclass'],
        "Education": row['Education'],
        "MaritalStatus": row['MaritalStatus'],
        "Occupation": row['Occupation'],
        "Relationship": row['Relationship'],
        "Race": row['Race'],
        "Salary": row['Salary'],
        "Sex": row['Sex'],
        "CapitalGain": row['CapitalGain'],
        "CapitalLoss": row['CapitalLoss'],
        "HoursPerWeek": row['HoursPerWeek'],
        "Country": row['Country']      
    }]
}