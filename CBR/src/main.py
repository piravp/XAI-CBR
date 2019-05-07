import requests
import helper

# https://www.openml.org/d/1590

class RESTApi:
    def __init__(self):
        pass

    # Create case-base
    # Case-base is created by providing a cb-name
    def addCaseBase(self, cbname):
        r = requests.put('http://localhost:8080/casebases/{}'.format(cbname))
        return r.status_code

    # Return all case-bases
    def getCaseBases(self):
        res = requests.get('http://localhost:8080/casebase')
        return helper.pprint(res.json())

    # Add concept
    # Concept is added by defining a conceptID(string) and doing a PUT call to the API
    def addConcept(self, conceptID):
        r = requests.put('http://localhost:8080/concepts/{}'.format(conceptID))
        return r.status_code

    # Add instance
    # An instance is created by providing conceptID, casebaseID and arbitrary caseID(string) and doing a PUT call
    def addInstance(self, conceptID, casebaseID, caseID, casedata):
        r = requests.put('http://localhost:8080/concepts/{}/casebases/{}/instances/{}' 
                .format(conceptID, casebaseID, caseID), data={'key': casedata})
        return r.status_code

    # Return all instances
    def getInstances(self, conceptID):
        res = requests.get('http://localhost:8080/concepts/{}/instances'.format(conceptID))
        return helper.pprint(res.json())

# print(addCaseBase('cb22'))
# print(getCaseBases())
# print(addConcept('People'))
# print(addInstance(conceptID='People', casebaseID='default', caseID='0', casedata='test'))
# print(getInstances(conceptID='Person'))


api = RESTApi()
api.__init__