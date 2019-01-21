from db import iris_collection as iris_db
from loaddata import iris_df_dict
class Case:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, iris_class):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.iris_class = iris_class


class CaseManager:
    def __init__(self, case: Case):
        self.case = case

    def add_case(self, case):
        # Case in dict-format
        pass


iris_db.insert_many(iris_df_dict.to_json())
# for key, value in iris_df_dict.items():
#     print(key, value)

