import numpy as np

# Keep track of the different attributes:
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


class Person():
    #['age' 'workclass' 'education' 'marital status' 'occupation'     
    # 'relationship' 'race' 'sex' 'capital gain' 'capital loss'                                                                                    'United-States 'relationship' 'race' 'sex' 'capital gain' 'capital loss'
    #'hours per week' 'country' 'income']
    def __init__(self, age:int, workclass:str, education:str, martial_status:str, occupation:str,
        relationship:str, race:str, sex:str, capital_gain:int, capital_loss:int,
        hours_per_week:int,country:str):
        self.age = age
        self.workclass = workclass
        self.education = education
        self.martial_status = martial_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.country = country
        self.income = None

    def transform(self): # transform input space into something the network can use.
        # Assumes all are strings.
        # first discretisize the age and hours_per_week
        pass
    def flatten(self):
        temp = [] # array to store the vars7
        print(dir(self))
        return np.array([self.age,self.workclass,self.education,
        self.martial_status,
        self.occupation,
        self.relationship,
        self.race,
        self.sex,
        self.capital_gain,
        self.capital_loss,
        self.hours_per_week,
        self.country])
        

    # Check that it is valid attributes

a = Person(10, 'Federal-gov','Dropout', 'Never-Married', 'Other','Own-child','Amer-Indian-Eskimo',
'Male','None','None',30,'British-Commonwealth')
