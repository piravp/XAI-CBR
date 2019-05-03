import uuid
from CBR.dataloader2 import Datamanager
from CBR.DBManager import DBManager

# INFO: To have more than one cb, create another case-base and call it case-base2 or something

class Case:
    def __init__(self, age, workclass, education, marital_status, occupation, relationship, race,
                    sex, capital_gain, capital_loss, hours_per_week, country, income):
        self.id = uuid.uuid4().hex

        # Attributes
        self.age = age
        self.workclass = workclass
        self.education = education
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
    def __init__(self):
        # self.case = case
        self.cases = {}
        self.containers = {}           # All case-bases
        self.db = DBManager()

    def retrieve_data(self, datamanager):
        """ Retrieve data from Datamanager class """
        cases = []
        for row in datamanager.ret.df_unencoded.itertuples(index=False):
            c = Case(age=row[0], workclass=row[1], education=row[2], marital_status=row[3], occupation=row[4],
                     relationship=row[5], race=row[6], sex=row[7], capital_gain=row[8], capital_loss=row[9],
                     hours_per_week=row[10], country=row[11], income=row[12])

            # self.add_case(case=c)
            cases.append(c)
            self.add_cases(cases=cases)


    def create_container(self, name, db=True):
        """ Create container

        Args:
            db: Add to db if flag is true

        """

        # Create new empty container
        self.containers[name] = {}

        if db:
            self.db.create_container_db(name)

    # ----------------------------------------------------------------------------------------------------- #
    #                                       C A S E  M A N A G E M E N T                                    #
    # ----------------------------------------------------------------------------------------------------- #

    def add_case(self, case, cb=None):
        """ Add _one_ case to a case-base """

        # No case-base exists
        if cb is None:
            cb = 'case-base'
            self.create_container(name=cb)

        # Case in dict-format
        self.cases[case.id] = case

        # Update case-base
        self.containers[cb].update(self.cases)
        print()


    def add_cases(self, cases, cb=None):
        """ Add multiple cases to the case-base (more time saving than calling add_case() multiple times) """

        # No case-base exists
        if cb is None:
            cb = 'case-base'
            self.create_container(name=cb)

        # Add case to dict
        for case in cases:
            # Case in dict-format
            self.cases[case.id] = case

        # Update case-base only after each case has been added to case dictionary
        self.containers[cb].update(self.cases)


    def delete_case(self, case):
        # Beware that this will result in a mutated dictionary
        del self.cases[case.id]

    # def update(self, container):
    #     """ Used to update containers with an updated case-base or knowledge-base
    #
    #     Args:
    #         container: Name of container
    #
    #     """
    #
    #     # TODO: Update self.cases against DB
    #     self.containers[container].update(self.cases)




dm = Datamanager(dataset='adults')

cbr = CBRManager()
cbr.create_container(name='test_contain', db=True)
# cbrman.retrieve_data(dm)
# cbrman.add_case(case)

