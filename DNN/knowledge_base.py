# Reposible for storing the knowledge base ( of explanatory knowledge)
# The data is stored in a json file.
import pathlib
from pathlib import Path
import json
from explanation import Explanation
import os
from collections import defaultdict
#import explanation
# Get path of parent nr 2, and append corresponding data path
folderpath = Path(__file__).parents[1]/"Data"/"knowledge_base"

# KnowledgeBase contains a list of explanation objects. 
# self.ID = 0 # ID of next explanation object.
# self.KB = {"id_0":<explanation>, "id_1":<explanation>}

class KnowledgeBase(json.JSONEncoder, json.JSONDecoder):
    # KnowledeBase is just a dictionary of explanations
    # TODO: get number of stored cases, for unique IDs.
    def __init__(self, file_name):
        if(not file_name.endswith(".json")): # if it doesn't end with .json
            file_name += ".json"

        self.name = file_name
        filepath = folderpath/self.name
        # file_name is where we will store the knowlede base.
        # Either create a new file, if file_name do not already exist
        if(not filepath.exists()):
            print("Nothing to load, no file")
            self.id = 0 # init ID to zero. 
            self.KB = defaultdict(Explanation) # empty json object

            # create file
        else: # Open the file
            with filepath.open(mode='r') as json_file:
                print("Loading from file", filepath)
                #data = json.load(json_file)
                if(self.load_json(json_file)):
                    print("Succesfully loaded KnowledeBase with size: {}, id: {}".format(len(self.KB),self.id))
                #data = json.load(fp=json_file,cls=decode_json)
                #data = json.load(json_file, object_hook=self.decode_knowledge_base)

    def add_knowledge(self,exp:Explanation):
        if(str(self.id) in self.KB.keys()): # If ID spot is not free.
            self.update_id() # simply need to figure out the next free number from our dict
        
        for k,e in self.KB.items():
            if(e == exp):
                return k
        self.KB[str(self.id)] = exp
            # if storage successfull, return ID, otherwise return error
        self.save() # save knowledge base
        return self.id # id that we need to store in the Case.

    def update_id(self): # TODO: use the IDS of each explanation as well, not just number of instances.
        keys = self.KB.keys() # simply need a number not in this list.
        while(str(self.id) in keys): # incremenet until no longer in list
            self.id += 1

    def save(self): 
        filepath = folderpath/self.name
        with filepath.open(mode='w') as json_file:
            json.dump(self, json_file, default=self.default, indent=4) # save file 

    def default(self,obj):
        #print("default",type(obj),obj.__dict__)
        if(isinstance(obj,KnowledgeBase)):
            return {
                "__class__":obj.__class__.__name__,
                "name":obj.name,
                "id":obj.id,
                "KB":obj.KB
            }
        elif(isinstance(obj,Explanation)):
            return obj.default(obj) # use default encoding from corresponding class.
        else: # Stop encoding. Wrong input types.
            raise ValueError("Cant encode class")
        # return obj.__dict__ Handle every class with every attribute.

    
    def load_json(self, dct): # dict -> python objects
        dct = json.load(dct) # read json file: str -> dict
        if("__class__" in dct):
            class_name = dct.pop("__class__")
            if(class_name == "KnowledgeBase"):
                self.name = dct["name"]
                self.id = int(dct["id"])
                self.KB = defaultdict(Explanation)
                # deal with KB
                KB = dct.pop("KB")
                for key, exp in KB.items():
                    class_name = exp.pop("__class__")
                    if(class_name == "Explanation"):
                        self.KB[key] = Explanation(**exp)
                return True
        return False

    def get(self, id, default=None): # return knowledge
        id = self.convert_id(id)
        if(id in self.KB):
            return self.KB.get(id)
        else: # Handle cases where knowledge is not present.
            return default

    def delete_knowledge(self,id,save=False): #Try to delete case with ID
        # simply pop the ID from dictionary and save.
        id = self.convert_id(id) # transform to string
        if(id in self.KB):
            self.KB.pop(id)
            self.save() # save to file.
        else:
            print("ID '{}' not in knowledge base".format(id))

    def convert_id(self,id):
        if(isinstance(id,int)):
            id = str(id)
        return id

def decode_json(dct,*kw):
    print(dct)

def test_knowledge_base():
    e1 = Explanation(feature=[1,2],names=[3,3],precision=[0.67,0.9],coverage=[0.2,0.05], prediction=1)
    e2 = Explanation(feature=[3,4],names=[4,6],precision=[0.8,0.99],coverage=[0.4,0.10], prediction=1)

    KB = KnowledgeBase("abc")

    KB.add_knowledge(e1)
    KB.add_knowledge(e2)
    print(KB.__dict__)
    KB.save()

def test_kb_load():
    KB = KnowledgeBase("ab")
    print(KB.__dict__)
    print(KB.name)
    for key, exp in KB.KB.items():
        print(key, exp,exp.features(1))

    print(KB.KB)
    print(KB.KB.items())
    print(KB.get("0"))
    print(KB.get(0))
    KB.delete_knowledge(8)
    print(KB.get(8))
    #print(KB.KB.get(0))
    #print(KB.KB[0])

# Final
def test_load_kb():
    KB = KnowledgeBase("aaab")
    print(KB.__dict__)
    print(KB.name)

def test_add_more():
    KB = KnowledgeBase("t_ab") # create empty knowledge base
    e1 = Explanation(feature=[1,2],names=[4,3],precision=[0.67,0.9],coverage=[0.2,0.05], prediction=0)
    e2 = Explanation(feature=[3,4],names=[2,2],precision=[0.8,0.99],coverage=[0.4,0.10], prediction=1)

    # print('e1',e1)

    KB.add_knowledge(e1)
    KB.add_knowledge(e2)

    print(KB.KB)

def test_similarity():
    e1 = Explanation(feature=[1,2],names=[4,3],precision=[0.67,0.9],coverage=[0.2,0.05], prediction=0)
    e2 = Explanation(feature=[3,4],names=[2,2],precision=[0.8,0.99],coverage=[0.4,0.10], prediction=1)
    e3 = Explanation(feature=[3,4],names=[2,2],precision=[0.8,0.99],coverage=[0.4,0.10], prediction=1)

    print(e1 == e2)
    print(e2 == e3)

def test_user_defined():
    KB = KnowledgeBase("test_kb") # create empty knowledge base
    print(KB.KB)
    ex1 = KB.get(0)
    print(ex1,ex1.user_defined)

def test_empty_kb():
    KB = KnowledgeBase("t_kb") # create empty knowledge base
    e1 = Explanation(feature=[1,2],names=[4,3],precision=[0.67,0.9],coverage=[0.2,0.05], prediction=0)
    print(type(e1),e1)
    KB.add_knowledge(e1)

#test_knowledge_base()
#test_knowledge_base_load()
# test_knowledge_base_save()
#test_kb_load()
#test_kb_load()
#test_knowledge_base()
# test_add_more()
#test_load_kb()
<<<<<<< HEAD
#test_user_defined()
#test_similarity()
test_empty_kb()
=======
# test_add_more()
>>>>>>> 424e0ee2fcdb2b32fed020579d0e5a5b48cd1aef
