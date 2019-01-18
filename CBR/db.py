import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["Case-Base"]

mycol = mydb["cases"]

mydict = { "classification": "1", "cancer": "1" }

x = mycol.insert_one(mydict)

print(myclient.list_database_names())