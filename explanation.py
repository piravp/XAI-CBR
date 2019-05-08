# explanation object, that will be stored in each case
from DNN.Induction.Anchor import anchor_explanation
import random
class Explanation(anchor_explanation.AnchorExplanation): # extend AnchorExplanation
    def __init__(self,id,names=None,feature=None,precision=None,coverage=None):
        self.id = id
        if(any([names,feature,precision,coverage])): # if ether of these have a value
            exp_map = {'feature': [], 'mean': [], 'precision': [],
                'coverage': [], 'examples': [], 'all_precision': 0}
            exp_map['names'] = names
            exp_map['feature'] = feature
            exp_map['precision'] = precision
            exp_map['coverage'] = coverage
            super(Explanation, self).__init__(type_='tabular',exp_map=exp_map,as_html=None)

        # TODO: get other value aswell 
    #def names(self,partial_index):
    #    self.super(Explanation, self).names(partial_index)




def test_explanation():
    import json
    e = Explanation(id=1,feature=[1,2],names=['test1','test2'],precision=[0.67,0.9],coverage=[0.2,0.05])

    print(e.features())
    print(e.names())
    print(e.coverage())
    print(e.precision())

    # Partial anchor explanation
    print(e.features(0))
    print(e.names(0))
    print(e.coverage(0))
    print(e.precision(0))
    explanation = {1:e.exp_map}
    print(e.exp_map)

    print(json.dumps(explanation))

test_explanation()