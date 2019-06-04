import ast
import json
from scipy import spatial

class Case(json.JSONEncoder):
    def __init__(self, age:int, workclass:str, education:str, martial_status:str, occupation:str,
        relationship:str, race:str, sex:str, capital_gain:int, capital_loss:int,
        hours_per_week:int,country:str, explanation:int, prediction:int, weight, similarity=None, caseID=None, KB = None):
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
        #age, workclass, education, martial_status, occupation, relationship, race, sex, 
        #               capital_gain, capital_loss, hours_per_week, country, prediction(salary)
        self.columns = { # TODO: decode ...
            0:self.age,
            1:self.workclass,
            2:self.education,
            3:self.martial_status,
            4:self.occupation,
            5:self.relationship,
            6:self.race,
            7:self.sex,
            8:self.capital_gain,
            9:self.capital_loss,
            10:self.hours_per_week,
            11:self.country
        }

        if(KB is not None):
            self.KB = KB # knowledge_base, that we keep the different explanations in.

    def check_equal(self,other): # check if a case is exactly the same as this one.
        if( self.age            == other.age            and
            self.workclass      == other.workclass      and
            self.education      == other.education      and
            self.martial_status == other.martial_status and
            self.occupation     == other.occupation     and
            self.race           == other.race           and
            self.sex            == other.sex            and
            self.capital_gain   == other.capital_gain   and
            self.capital_loss   == other.capital_loss   and
            self.hours_per_week == other.hours_per_week and
            self.country        == other.country):
            return True
        return False


    def checkCosineDistance(self,other):
        return spatial.distance.cosine(self.weight, other.weight)

    def checkEuclidianDistance(self,other):
        return spatial.distance.euclidean(self.weight, other.weight)

    def checkCosinePrediction(self,other): # if wrong prediction, no similarity
        if(self.prediction == other.prediction):
            return spatial.distance.cosine(self.weight, other.weight)
        else:
            return 1

    # We want to figure out if the partial explanation holds related to its own.
    def checkSimilarityPartialExplanation(self, other, KB_s, KB_o): 
        if(self.prediction != other.prediction): # if not even similar prediction, we might as well return with a low similarity score.
            return 0 # return -1, as in the prediction is not even correct.
        # Check the similarity between this case and another.
        # Return partial or exact match between anchors.
        exp_s = KB_s.get(self.explanation)
        exp_o = KB_o.get(other.explanation)

        partial_fit = 0

        limit = len(exp_s.features()) # don't want to overeach
        for exp in range(len(exp_o.features())): # if partial explanation fit, we can use it for generating an explanation.
            if(exp>= limit):
                print(exp_o.features(exp))
            if(exp_o.check_similarity(exp_s)):
                partial_fit = exp

        return partial_fit

    def checkAnchorFitting(self, other, KB_o, decoder_f, decoder_v):
        # check if the others explanation fit on our case.
        if(self.prediction != other.prediction): # if not even similar prediction, we might as well return with a low similarity score.
            return 0 # return -1, as in the prediction is not even correct
        #check if explanation fits
        exp_o = KB_o.get(other.explanation)

        partial_fit = 0
        for p in range(len(exp_o.features())): # if partial explanation fit, we can use it for generating an explanation.
            f = exp_o.feature(p)[p] # partial index and index.
            v = exp_o.names(p)[p]  
            
            f_dec = decoder_f[f] # get freature f
            v_dec = decoder_v[f][v] # from feature f, get value v

            if(self.columns[f] != v_dec):
                return partial_fit
            else:
                partial_fit += 1

        return partial_fit


    def isSolved(self):
        # Simply check whether or not this particular case has a solution
        if(self.explanation is None):
            return False
        return True

    # pylint: disable=E0202
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
                "Prediction":     str(o.prediction),
                "Race":           o.race,
                "Relationship":   o.relationship,
                "Sex":            o.sex,
                "Weight":         str(o.weight),
                "Workclass":      o.workclass
        }

    def __str__(self):
        return json.dumps(Case.default(self))

