import json

age = 0 # int discretized
workclass = {'Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'}
education = {'Associates', 'Bachelors', 'Doctorate', 'Dropout', 'High School grad', 'Masters', 'Prof-School'}
marital_status = {'Married', 'Never-Married', 'Separated', 'Widowed'}
occupation = {'Admin', 'Blue-Collar', 'Military', 'Other', 'Professional', 'Sales', 'Service', 'White-Collar'}
relationship = {'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'}
race = {'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'}
sex = {'Female', 'Male'}
capital_gain = {2:'High', 1:'Low', 0:'None'} # int
capital_loss = {2:'High', 1:'Low', 0:'None'} # int
hours_per_week = 0 #int discretized
country = {'British-Commonwealth', 'China', 'Euro_east', 'Euro_south', 'Euro_west', 'Latin-America', 'Other', 'SE-Asia', 'South-America', 'United-States'}


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