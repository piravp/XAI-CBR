# from db import iris_collection as iris_db
# from cbr.dataloader import iris_df_dict
import uuid

# "age", "workclass", "fnlwgt", "education",
# "education-num", "marital_status", "occupation",
# "relationship", "race", "sex", "capital gain",
# "capital_loss", "hours per week", "country", 'income'

class Case:
    def __init__(self, age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race,
                    sex, capital_gain, capital_loss, hours_per_week, country, income):
        self.id = uuid.uuid4().hex

        # Attributes
        self.age = age
        self.workclass = workclass
        # self.fnlwgt = fnlwgt
        self.education = education
        # self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hpw = hours_per_week
        self.country = country
        self.income = income



class CBRManager:
    def __init__(self, case: Case):
        # self.case = case
        self.cases = {}
        self.cb_idx = 0
        self.all_cb = {}           # All case-bases

    # Run after each operation that modifies the case-base
    def update(self):
        # TODO: Update self.cases against DB
        pass

    # Create case-base
    def create_cb(self, name):
        # Create new empty case-base

        self.all_cb[name] = {}

    # ---------- Case Management --------------
    # Add a case to given case-base
    def add_case(self, case, cb):
        # Case in dict-format
        self.cases[case.id] = case
        # pass

    def delete_case(self, case):
        # Beware that this will result in a mutated dictionary
        del self.cases[case.id]
        # pass


from CBR.dataloader2 import *

dm = Datamanager(dataset='adults')
df = dm.ret.df_unencoded
print(df)



