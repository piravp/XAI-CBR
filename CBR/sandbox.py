# class Case:
#     def __init__(self, sepal_length, sepal_width, petal_length, petal_width, iris_class):
#         self.sepal_length = sepal_length
#         self.sepal_width = sepal_width
#         self.petal_length = petal_length
#         self.petal_width = petal_width
#         self.iris_class = iris_class
class Case:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, predicted_class):
        self.id = None
        # Attributes
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        # Predicted class
        self.predicted_class = predicted_class
        # Problem, solution and expl
        self.problem = None
        self.solution = None
        self.explanation = None

        # Et system for å vekte de ulike attributene forskjellig
        # Vekter - target vektes høyere

    def __str__(self):
        return 'id : {}'.format(self.id)

    def __iter__(self):
        return iter(self)

    def __next__(self):
        pass
case1 = Case(sepal_length=5.4, sepal_width=3.9, petal_length=1.1, petal_width=0.3, predicted_class=None)
# iter_list = case1.__iter__()
print(case1.__dict__)
# cc = dict(enumerate(case1))
temp = 0 + 1

