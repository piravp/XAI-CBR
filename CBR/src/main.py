import requests
import helper
import json
import pandas as pd
import matplotlib
import seaborn as sb
import matplotlib.pyplot as plt

# https://www.openml.org/d/1590


class RESTApi:
    def __init__(self):
        pass

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
    # Add instance
    # An instance is created by providing conceptID, casebaseID and arbitrary caseID(string) and doing a PUT call
    def addInstance(self, casebaseID, conceptID, caseID, casedata):
        r = requests.put('http://localhost:8080/concepts/{}/casebases/{}/instances/{}' 
                .format(conceptID, casebaseID, caseID), data={'key': casedata})
        return r.status_code


    def addInstancesJSON(self, casebaseID, conceptID, cases):
        r = requests.post(url='http://localhost:8080/concepts/{}/casebases/{}/instances' 
                .format(conceptID, casebaseID), params={"cases" : json.dumps(cases)})
        return r.text
        # return 'http://localhost:8080/concepts/{}/casebases/{}/instances?{}'.format(conceptID, casebaseID, cases)


    # Return instances for one specific case-base
    def getAllInstancesInCaseBase(self, conceptID, casebaseID):
        res = requests.get('http://localhost:8080/concepts/{}/casebases/{}/instances'.format(conceptID, casebaseID))
        raw = pd.DataFrame(res.json())
        instances = raw.apply(pd.to_numeric, errors='coerce').fillna(raw)
        return instances#helper.pprint(res.json())

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

    # ----------------------------------------------------------------------------- #
    #                               Similarity                                      #
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

api = RESTApi()


# res = api.addInstancesJSON(casebaseID='cb0', conceptID='Person', cases={"cases":[{"Age":22, "Gender":"Male"}]})
# res = api.getInstances(conceptID='Person')
# res = api.getAllInstancesInCaseBase(conceptID='Person', casebaseID='cb0')
# res = api.getAttributes('Person')
# res = api.addAttribute(conceptID='Person', attrName='Education', attrJSON={"type": "Symbol", "allowedValues": ["High school", "Bachelor", "Master"]})
res = api.retrieve_k_sim_byID(conceptID='Person', casebaseID='cb0', queryID='Person-cb018', k=5)
print(res)
api.plot_retrieve_k_sim_byID(res)





