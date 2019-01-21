import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

# Connect to db
mydb = myclient["Case-Base"]

# Create collection
# case_collection = mydb["cases"]
iris_collection = mydb["irises"]

# case = { "classification": "1", "cancer": "1", "malignant": "no" }
#
# # Add entries to collection
# x = case_collection.insert_one(case)
# print(myclient.list_database_names())