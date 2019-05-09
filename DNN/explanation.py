# explanation object, that will be stored in each case
from Induction.Anchor import anchor_explanation
import random
import json
class Explanation(anchor_explanation.AnchorExplanation, json.JSONEncoder): # extend AnchorExplanation
    def __init__(self,names=None,features=None,precision=None,coverage=None):
        print(names,features,precision,coverage)
        if(not all([names,features,precision,coverage])): # if ether of these have a value
            raise ValueError("Need input: names, features, precision and coverage")
        else:
            exp_map = {'features': [], 'mean': [], 'precision': [],
                'coverage': [], 'examples': [], 'all_precision': 0} # Initial mapping
            
            # Check if values are correct length with eachother
            if(len(names) != len(features)):
                #raise ValueError("Length of the input 'names' {names} are not the same as 'feature' {feature} ",names,feature)
                msg = "Length of the input 'names' {names} are not the same as 'features' {features} ".format(names,features)
                raise ValueError(msg)
            elif (len(coverage) != len(precision)):
                msg = "Length of the input 'coverage' {coverage} are not the same as 'precision' {precision}".format(coverage,precision)
                raise ValueError("Length of the input 'coverage' are not the same as 'precision'")

            exp_map['names'] = names
            exp_map['feature'] = features
            exp_map['precision'] = precision
            exp_map['coverage'] = coverage
            super(Explanation, self).__init__(type_='tabular', exp_map=exp_map, as_html=None)

    def __str__(self): 
        return str(self.exp_map)
        
        # TODO: get other value aswell 
    def default(self, obj): # no need to store whole dictionary.
        return {"__class__":obj.__class__.__name__,
                "names":self.exp_map["names"],
                "features":self.exp_map['feature'],
                "precision":self.exp_map['precision'],
                "coverage":self.exp_map['coverage']
                }


def test_explanation():
    import json
    e = Explanation(feature=[1,2],names=['test1','test2'],precision=[0.67,0.9],coverage=[0.2,0.05])

    print(e.features())
    print(e.names())
    print(e.coverage())
    print(e.precision())

    ef = Explanation(feature=[3,5],names=['test1','test2'],precision=[0.67,0.9],coverage=[0.2,0.05])

    # Partial anchor explanation
    print(e.features(0))
    print(e.names(0))
    print(e.coverage(0))
    print(e.precision(0))
    explanation = {1:e.exp_map}
    print(e.exp_map)

    print(json.dumps(explanation))

#test_explanation()