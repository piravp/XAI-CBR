import pymongo



class DBManager():
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")      # Create connection to mongo-client
        self.db = self.client.xai_cbr                                        # Database


    def add_instance_db(self, instance, container="case-base"):
        container = self.db[container]
        container.insert_one(instance)

    def remove_instance_db(self):
        pass


    def create_container_db(self, name):
        if name in self.db.list_collection_names():
            print('A container named \'{}\' already exists.'.format(name))
            return

        # Create container
        # need to insert a dummy element to instantiate a collection in mongodb
        # self.containers[name] = self.db[name]
        new_container = self.db[name]
        new_container.insert_one({ "dummykey": "dummyvalue" })
        new_container.delete_many({})






# mydict = { "classification": "1", "cancer": "1" }               # One case-entry

# print(client.list_database_names())

# # Create collection
# # case_collection = mydb["cases"]
# iris_collection = mydb["irises"]
#
