import requests
import helper
import json
import pandas as pd

# https://www.openml.org/d/1590

baseurl = "http://localhost:8080/"

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



api = RESTApi()


# res = api.addInstancesJSON(casebaseID='cb0', conceptID='Person', cases={"cases":[{"Age":22, "Gender":"Male"}]})
# res = api.getInstances(conceptID='Person')
res = api.getAllInstancesInCaseBase(conceptID='Person', casebaseID='cb0')

print(res)



# print(addCaseBase('cb22'))
# print(getCaseBases())
# print(addConcept('People'))
# print(addInstance(conceptID='People', casebaseID='default', caseID='0', casedata='test'))
# print(getInstances(conceptID='Person'))

