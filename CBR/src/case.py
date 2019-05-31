import ast
import json
class Case(json.JSONEncoder):
    def __init__(self, age:int, workclass:str, education:str, martial_status:str, occupation:str,
        relationship:str, race:str, sex:str, capital_gain:int, capital_loss:int,
        hours_per_week:int,country:str, explanation:int, prediction:int, weight, similarity=None, caseID=None):
        # column index: age, workclass, education, martial_status, occupation, relationship, race, sex, 
        #               capital_gain, capital_loss, hours_per_week, country, prediction(salary)
        #self.column_index = ()
        self.similarity = similarity

        self.age = age                      # integer <0,+>
        self.workclass = workclass          # 'Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc',
                                            #+'State-gov', 'Without-pay'
        self.education = education          # 'Associates', 'Bachelors', 'Doctorate', 'Dropout', 'High School grad', 
                                            #+'Masters', 'Prof-School'
        self.martial_status = martial_status# 'Married', 'Never-Married', 'Separated', 'Widowed'
        self.occupation = occupation        # 'Admin', 'Blue-Collar', 'Military', 'Other', 'Professional', 'Sales', 
                                            #+'Service', 'White-Collar'
        self.relationship = relationship    # 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
        self.race = race                    # 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
        self.sex = sex                      # 'Female', 'Male'
        self.capital_gain = capital_gain    # 'None', 'Medium', 'High'
        self.capital_loss = capital_loss    # 'None', 'Medium', 'High'
        self.hours_per_week=hours_per_week  # integer <0,+>
        self.country = country              # 'British-Commonwealth', 'China', 'Euro_east', 'Euro_south', 'Euro_west',
                                            #+'Latin-America', 'Other', 'SE-Asia', 'South-America', 'United-States'
        self.prediction = prediction        # 0, 1 [ < 50 000,  >=50 000]
        if(isinstance(weight, str)):
            # Need to map the list from "[a,b,c,]" to [a,b,c]
            self.weight = ast.literal_eval(weight)
        else:
            self.weight = weight                # [a,b,c,d,e,f,g,h,i,j] 12 weights, one per attribute. age - country
        # Soulution part
        self.explanation = explanation      # id pointer to knowledge base.
        # TODO: Turn String to int values.

    def checkSimilarity(self, other):
        # Check the similarity between this case and another.
        # Simply check wheter or not the explanation fits
        pass

    @staticmethod
    def default(o):
        return {
                "Age":            o.age,
                "CapitalGain":    o.capital_gain,
                "CapitalLoss":    o.capital_loss,
                "Country":        o.country,
                "Education":      o.education, 
                "Explanation":    o.explanation,
                "HoursPerWeek":   o.hours_per_week,
                "MaritalStatus":  o.martial_status,
                "Occupation":     o.occupation,
                "Prediction":     o.prediction, 
                "Race":           o.race,
                "Relationship":   o.relationship,
                "Sex":            o.sex,
                "Weight":         str(o.weight),
                "Workclass":      o.workclass
        }

    def __str__(self):
        return json.dumps(Case.default(self))

