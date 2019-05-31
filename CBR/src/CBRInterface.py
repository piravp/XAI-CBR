import requests
import helper
import json
import pandas as pd
import matplotlib
import seaborn as sb
import matplotlib.pyplot as plt
from CBR.src.case import Case
# https://www.openml.org/d/1590

class RESTApi:
    " Functions as an interface to the REST api "
    def __init__(self):
        pass

    def checkStatus(self):  # Assumes that the MyCBR project have a casebase.
        try:
            res = requests.get('http://localhost:8080/casebase')
            return res.status_code # 200 is success
        except Exception: # Anything else is either server not ready or 404 missing. 
            return 500

    # ----------------------------------------------------------------------------- #
    #                               Case-Base                                       #
    # ----------------------------------------------------------------------------- #
    # Create case-base
    # Case-base is created by providing a cb-name
    def createCaseBase(self, cbname):
        r = requests.put('http://localhost:8080/casebases/{}'.format(cbname))
        return r.status_code

    # Return all case-bases
    def getCaseBases(self):
        res = requests.get('http://localhost:8080/casebase')
        return helper.pprint(res.json())

    def getCaseBaseID(self):
        r = requests.get('http://localhost:8080/casebase')
        r_json = r.json() # resulting json
        if(len(r_json["caseBases"]) != 1):
            raise ValueError("More than one CaseBase in the project")
        return r_json["caseBases"][0] # return the first concept ID

    # ----------------------------------------------------------------------------- #
    #                                Concept                                        #
    # ----------------------------------------------------------------------------- #
    # Create concept
    # Concept is created by defining a conceptID(string) and doing a PUT call to the API
    def createConcept(self, conceptID):
        r = requests.put('http://localhost:8080/concepts/{}'.format(conceptID))
        return r.status_code

    # Get all concepts
    def getConcepts(self):
        r = requests.get('http://localhost:8080/concepts')
        return helper.pprint(r.json())

    def getConceptID(self):
        r = requests.get('http://localhost:8080/concepts')
        r_json = r.json() # resulting json
        if(len(r_json["concept"]) != 1):
            raise ValueError("More than one concept in the CaseBase")
        return r_json["concept"][0] # return the first concept ID


    # Get all algamationfunctions
    def getAlgamationFunctions(self,conceptID):
        #http://localhost:8080/concepts/Person/amalgamationFunctions
        r = requests.get('http://localhost:8080/concepts/{}/amalgamationFunctions'.format(conceptID))
        r_json = r.json() # resulting JSON.
        return r_json["amalgamationFunctions"] # return list of algamation functions.

    # TODO: Add support for adding attributes with more types (integer, etc.)

    def addAttribute(self, conceptID, attrName, attrJSON):
        r = requests.put('http://localhost:8080/concepts/{}/attributes/{}'
                .format(conceptID, attrName), params={'attributeJSON' : json.dumps(attrJSON)})
        
        return r.text

    # Get all attributes
    def getAttributes(self, conceptID):
        r = requests.get('http://localhost:8080/concepts/{}/attributes'.format(conceptID))
        return helper.pprint(r.json())
    

    # ----------------------------------------------------------------------------- #
    #                               Instances                                       #
    # ----------------------------------------------------------------------------- #
    # Add one instance
    # An instance is created by providing conceptID, casebaseID and arbitrary caseID(string) and doing a PUT call
    # NOTE: THIS DOES NOT WORK, USE addInstancesJson() INSTEAD
    # def addInstance(self, casebaseID, conceptID, caseID, casedata):
    #     r = requests.put('http://localhost:8080/concepts/{}/casebases/{}/instances/{}' 
    #             .format(conceptID, casebaseID, caseID), data={'key': casedata})
    #     return r.status_code

    # Add several instances
    # An instance is created by providing conceptID, casebaseID and arbitrary caseID(string) and doing a PUT call
    def addInstancesJSON(self, casebaseID, conceptID, cases):
        print({"cases" : json.dumps(cases)})

        r = requests.post(url='http://localhost:8080/concepts/{}/casebases/{}/instances' 
                .format(conceptID, casebaseID), params={"cases" : json.dumps(cases)})
        return r.text
        # return 'http://localhost:8080/concepts/{}/casebases/{}/instances?{}'.format(conceptID, casebaseID, cases)

    def addInstancesCases(self, casebaseID, conceptID, cases):

        # dumped = json.dumps(cases, default=Case.default)
        # dumped = json.dumps(cases, default=Case.default)
        caseAsJson = {
            "cases" : cases
        }
        # dumped = {'cases' : dumped}
        # print(dumped)
        # print(json.dumps(cases, default=Case.default))

        r = requests.post(url='http://localhost:8080/concepts/{}/casebases/{}/instances' 
                .format(conceptID, casebaseID), params={"cases" : json.dumps(caseAsJson, default=Case.default)})
        return r.text
        
    # Return cases for one specific case-base
    def getAllInstancesInCaseBase(self, conceptID, casebaseID):
        res = requests.get('http://localhost:8080/concepts/{}/casebases/{}/instances'.format(conceptID, casebaseID))
        raw = pd.DataFrame(res.json())
        instances = raw.apply(pd.to_numeric, errors='coerce').fillna(raw)
        return instances#helper.pprint(res.json())

    # Return # of cases for one specific case-base
    def getCaseBaseSize(self, conceptID, casebaseID):
        res = self.getAllInstancesInCaseBase(conceptID, casebaseID)
        return res.shape[0]

    # Return ALL instances
    def getInstances(self, conceptID):
        res = requests.get('http://localhost:8080/concepts/{}/instances'.format(conceptID))
        # df = pd.DataFrame()
        return helper.pprint(res.json())

    # Delete one case
    def deleteInstance(self, casebaseID, conceptID, instanceID):
        res = requests.delete('http://localhost:8080/concepts/{}/casebases/{}/instances/{}'
                    .format(conceptID, casebaseID, instanceID))
        return res.text

    # Delete ALL instances
    def deleteAllInstances(self, conceptID):
        res = requests.get('http://localhost:8080/concepts/{}/instances'.format(conceptID))
        return res.text

    # Change one (existing) attribute for one case
    def modifyAttributeInCase(self, conceptID, casebaseID, caseID, attributeName, value):
        # Retrieve case
        res = self.getAllInstancesInCaseBase(conceptID=conceptID, casebaseID=casebaseID)
        idx = res[res['caseID']==caseID].index.item()                                               # index of case in df that matches the caseID we're looking for
        row = res.iloc[[idx]]                                                                       # find in df
        row = row.to_json(orient='records').replace('[','').replace(']','')                         # convert to json (string)
        row = json.loads(row)                                                                       # convert to json (object/dict)

        # Delete previous case
        if eval(self.deleteInstance(casebaseID=casebaseID, conceptID=conceptID, instanceID=caseID).capitalize()):
            print('\'{}\' was deleted...'.format(caseID))
            # Create new case if previous version was successfully deleted
            row[attributeName] = value                                                              # change value
            case = helper.caseAsJson(row)                                                           # convert to format accepted by REST
            r = self.addInstancesJSON(casebaseID=casebaseID, conceptID=conceptID, cases=case)       # add instance
            return r

        return False

    # ----------------------------------------------------------------------------- #
    #                           Similarity retrieval                                #
    # ----------------------------------------------------------------------------- #
    def retrieve_k_sim_byID(self, conceptID, casebaseID, queryID, k):
        res = requests.get('http://localhost:8080/concepts/{}/casebases/{}/retrievalByID?caseID={}&k={}'
                    .format(conceptID, casebaseID, queryID, k))
        raw = pd.DataFrame(res.json())
        results = raw.apply(pd.to_numeric, errors='coerce').fillna(raw).sort_values(by='similarCases', ascending=False)
        return results

    def plot_retrieve_k_sim_byID(self, data):
        plt.xticks(rotation=35)
        ax = sb.barplot(x=data.index, y="similarCases", data=data)
        plt.show()

# api = RESTApi()
# res = api.addInstancesJSON(casebaseID='cb0', conceptID='Person', cases={"cases":[{"Age":11}]})
# res = api.getInstances(conceptID='Person')
# res = api.getAttributes('Person')
# res = api.addAttribute(conceptID='Person', attrName='Education', attrJSON={"type": "Symbol", "allowedValues": ["High school", "Bachelor", "Master"]})
# res = api.retrieve_k_sim_byID(conceptID='Person', casebaseID='cb0', queryID='Person-cb018', k=5)
# res = api.getAllInstancesInCaseBase(conceptID='Person', casebaseID='cb0')
# res = api.deleteInstance(casebaseID='cb0', conceptID='Person', instanceID='Person-cb03')
# res = api.getCaseBaseSize(conceptID='Person', casebaseID='cb0')
# res = api.modifyAttributeInCase(casebaseID='cb0', conceptID='Person', caseID='Person-cb08', attributeName='Age', value=25)

# print(res)
# api.plot_retrieve_k_sim_byID(res)





