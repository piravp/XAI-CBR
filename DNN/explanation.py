# explanation object, that will be stored in each case
from Induction.Anchor import anchor_explanation
import random
import json
class Explanation(anchor_explanation.AnchorExplanation, json.JSONEncoder): # extend AnchorExplanation
    def __init__(self,value=None,names=None,feature=None,precision=None,coverage=None,**args):
        if(not all([names,feature,precision,coverage])): # if ether of these have a value
            raise ValueError("Need input: names, feature, precision and coverage")
        else:
            exp_map = {'feature': [], 'mean': [], 'precision': [],
                'coverage': [], 'examples': [], 'all_precision': 0} # Initial mapping
            
            # Check if values are correct length with eachother
            if(len(names) != len(feature)):
                #raise ValueError("Length of the input 'names' {names} are not the same as 'feature' {feature} ",names,feature)
                msg = "Length of the input 'names' {}({}) are not the same as 'feature' {}({})".format(
                    names,len(names),feature,len(features))
                raise ValueError(msg)
            elif (len(coverage) != len(precision)):
                msg = "Length of the input 'coverage' {}({}) are not the same as 'precision' {}({})".format(
                    coverage,len(coverage),precision,len(precision))
                raise ValueError("Length of the input 'coverage' are not the same as 'precision'")
            # TODO: change names list to values from instance
            if(value is not None): #  we need to swap names list by these numbers instead.
                if(isinstance(value,list)):
                    names = value # this is the one we ultimatly want to store.
                
                print("value",value,names)

            # Init input
            exp_map['names'] = names
            exp_map['feature'] = feature
            exp_map['precision'] = precision
            exp_map['coverage'] = coverage
            super(Explanation, self).__init__(type_='tabular', exp_map=exp_map, as_html=None)

    def get_explanation(self,decoder_f,decoder_v, partial_index=None):
        # return string: "a"="1" AND "b"="2"
        feature = self.features(partial_index)
        names = self.names(partial_index)
        tmp = []
        for i,f in enumerate(feature): # f is the feature index, name[i] value index
            tmp.append("{} = {}".format(decoder_f[f], decoder_v[f][names[i]]))
        return ' AND '.join(tmp)


    def __str__(self): 
        return str(self.exp_map)
        
    def default(self, obj): # no need to store whole dictionary.
        return {"__class__":obj.__class__.__name__,
                "names":self.exp_map["names"],
                "feature":self.exp_map['feature'],
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