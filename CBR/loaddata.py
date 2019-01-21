import pandas
import json
iris_df = pandas.read_csv('iris.csv', names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class'])
iris_df.index.name = 'idx'
iris_df_dict = iris_df.to_dict(orient='index')

# print(iris_df.head(5))
iris_json = iris_df.to_json(orient='index')
parsed = json.loads(iris_json)
print(json.dumps(parsed, indent=4, sort_keys=True))


# https://wellsr.com/python/pandas-dataframe-from-dictionary-list-and-more/