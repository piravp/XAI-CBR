# XAI-CBR

## Set up MongoDB locally

Install `pymongo` package using pip. In addition, install RoboMongo (desktop app) for a nice UI to more easily view the contents of the db. Then run this to set up a db:
```python
import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["Case-Base"]
mycol = mydb["cases"]
mydict = { "classification": "1", "cancer": "1" }
x = mycol.insert_one(mydict)
print(myclient.list_database_names())
```

To view db inside RoboMongo:  
`File > Connect > Create > Name='Case-Base' > Save`