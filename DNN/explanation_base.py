# Reposible for storing the explanation base ( of explanatory knowledge), each knowledge instance is an explanation object.
# The data is stored in a json file.
import pathlib
from pathlib import Path
import json
import DNN
from explanation import Explanation
import os
from collections import defaultdict
#import explanation
# Get path of parent nr 2, and append corresponding data path
folderpath = Path(__file__).parents[1]/"Data"/"explanation_base"

# ExplanationBase contains a list of explanation objects. 
# self.ID = 0 # ID of next explanation object.
# self.EB = {"id_0":<explanation>, "id_1":<explanation>}

class ExplanationBase(json.JSONEncoder, json.JSONDecoder):
    # KnowledeBase is just a dictionary of explanations
    # TODO: get number of stored cases, for unique IDs.
    def __init__(self, file_name):
        if(not file_name.endswith(".json")): # if it doesn't end with .json
            file_name += ".json"

        self.name = file_name
        filepath = folderpath/self.name

        # Check if folder exists
        if(not folderpath.exists()):
            os.makedirs(folderpath)

        # file_name is where we will store the knowlede base.
        # Either create a new file, if file_name do not already exist
        if(not filepath.exists()):
            print("Nothing to load, no file, creating new empty Explanation Base")
            self.id = 0 # init ID to zero. 
            self.EB = defaultdict(Explanation) # empty json object

            # create file
        else: # Open the file
            with filepath.open(mode='r') as json_file:
                print("Loading from file", filepath)
                #data = json.load(json_file)
                if(self.load_json(json_file)):
                    print("Succesfully loaded ExplanationBase with size: {}, id: {}".format(len(self.EB),self.id))

    def add_knowledge(self,exp:Explanation):
        if(str(self.id) in self.EB.keys()): # If ID spot is not free.
            self.update_id() # simply need to figure out the next free number from our dict
        
        # Check if the explanation generated is exactly the same as a previous explanation in the EB.
        # This is when the Anchors are the same, and the coverage/precion scores are the same.
        for k,e in self.EB.items():
            if(e == exp):
                return str(k) # return key k, which is also the ID.
        self.EB[str(self.id)] = exp
            # if storage successfull, return ID, otherwise return error
        self.save() # save explanation base
        return str(self.id) # id that we need to store in the Case.

    def update_id(self): # TODO: use the IDS of each explanation as well, not just number of instances.
        keys = self.EB.keys() # simply need a number not in this list.
        while(str(self.id) in keys): # incremenet until no longer in list
            self.id = int(self.id) + 1

    def save(self): 
        filepath = folderpath/self.name
        with filepath.open(mode='w') as json_file:
            json.dump(self, json_file, default=self.default, indent=4) # save file 

    def default(self,obj):
        #print("default",type(obj),obj.__dict__)
        if(isinstance(obj, ExplanationBase)):
            return {
                "__class__":obj.__class__.__name__,
                "name":obj.name,
                "id":obj.id,
                "EB":obj.EB
            }
        elif(isinstance(obj, Explanation) or isinstance(obj, DNN.explanation.Explanation)):
            return obj.default(obj) # use default encoding from corresponding class.
        else: # Stop encoding. Wrong input types.
        #    print(obj)
        #    print(obj.name)
            raise ValueError("Cant encode class", type(obj))
        # return obj.__dict__ Handle every class with every attribute.
        #elif(obj.__class__.__name__() == "Explanation"):
        #    return obj.default(obj)
        #else:
        #    raise ValueError("Cant encode class", type(obj))


    def load_json(self, dct): # dict -> python objects
        # __class__: explanation
        #   EB:
        #       __class__:explanation
        dct = json.load(dct) # read json file: str -> dict
        if("__class__" in dct):
            class_name = dct.pop("__class__")
            if(class_name ==  "ExplanationBase"):
                self.name = dct["name"]
                self.id = int(dct["id"])
                self.EB = defaultdict(Explanation)
                # deal with EB
                EB = dct.pop("EB")
                for key, exp in EB.items():
                    class_name = exp.pop("__class__")
                    # deal with explanation
                    if(class_name == "Explanation"):
                        self.EB[key] = Explanation(**exp)
                return True
        return False

    def get(self, id, default=None): # return knowledge
        id = self.convert_id(id)
        if(id in self.EB):
            return self.EB.get(id)
        else: # Handle cases where knowledge is not present.
            return default

    def get_user_knowledge(self): # simply return all the knowledge assosiated with the user.
        user_knowledge = []
        for key, knowledge in self.EB.items():
            if(knowledge.user_defined):
                user_knowledge.append(knowledge)
        return knowledge

    def delete_knowledge(self,id,save=False): #Try to delete case with ID
        # simply pop the ID from dictionary and save.
        id = self.convert_id(id) # transform to string
        if(id in self.EB):
            self.EB.pop(id)
            self.save() # save to file.
        else:
            print("ID '{}' not in explanation base".format(id))

    def reset_knowledge(self):
        # simply reset knowledge base dictionary.
        self.id = self.convert_id(0) # back to 0
        self.EB = defaultdict(Explanation) # empty json object
        self.save() # overwrite the old knowledge-base


    def convert_id(self,id):
        if(isinstance(id,int)):
            id = str(id)
        return id

def decode_json(dct,*kw):
    print(dct)

def test_knowledge_base():
    e1 = Explanation(feature=[1,2],names=[3,3],precision=[0.67,0.9],coverage=[0.2,0.05], prediction=1)
    e2 = Explanation(feature=[3,4],names=[4,6],precision=[0.8,0.99],coverage=[0.4,0.10], prediction=1)

    EB = ExplanationBase("abc")

    EB.add_knowledge(e1)
    EB.add_knowledge(e2)
    print(EB.__dict__)
    EB.save()

def test_EB_load():
    EB = ExplanationBase("ab")
    print(EB.__dict__)
    print(EB.name)
    for key, exp in EB.EB.items():
        print(key, exp,exp.features(1))

    print(EB.EB)
    print(EB.EB.items())
    print(EB.get("0"))
    print(EB.get(0))
    EB.delete_knowledge(8)
    print(EB.get(8))
    #print(EB.EB.get(0))
    #print(EB.EB[0])

# Final
def test_load_EB():
    EB = ExplanationBase("aaab")
    print(EB.__dict__)
    print(EB.name)

def test_add_more():
    EB = ExplanationBase("t_ab") # create empty knowledge base
    e1 = Explanation(feature=[1,2],names=[4,3],precision=[0.67,0.9],coverage=[0.2,0.05], prediction=0)
    e2 = Explanation(feature=[3,4],names=[2,2],precision=[0.8,0.99],coverage=[0.4,0.10], prediction=1)

    # print('e1',e1)

    EB.add_knowledge(e1)
    EB.add_knowledge(e2)

    print(EB.EB)

def test_similarity():
    e1 = Explanation(feature=[1,2],names=[4,3],precision=[0.67,0.9],coverage=[0.2,0.05], prediction=0)
    e2 = Explanation(feature=[3,4],names=[2,2],precision=[0.8,0.99],coverage=[0.4,0.10], prediction=1)
    e3 = Explanation(feature=[3,4],names=[2,2],precision=[0.8,0.99],coverage=[0.4,0.10], prediction=1)

    print(e1 == e2)
    print(e2 == e3)

def test_user_defined():
    EB = ExplanationBase("test_EB") # create empty knowledge base
    print(EB.EB)
    ex1 = EB.get(0)
    print(ex1,ex1.user_defined)

def test_empty_EB():
    EB = ExplanationBase("t_EB") # create empty knowledge base
    e1 = Explanation(feature=[1,2],names=[4,3],precision=[0.67,0.9],coverage=[0.2,0.05], prediction=0)
    print(type(e1),e1)
    EB.add_knowledge(e1)

