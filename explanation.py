# explanation object, that will be stored in each case
import DNN.Induction.Anchor as Anchor

print(Anchor)
#print(Anchor)
#print(Anchor.anchor_explanation)
class Explanation(anchor_explation.AnchorExplanation): # extend AnchorExplanation
    def __init__(self,names=None,feature=None,precision=None,coverage=None):
        if(any([names,features,precision,coverage])): # if ether of these have a value
            self.exp_map['names'] = names
            self.exp_map['feature'] = feature
            self.exp_map['precision'] = precision
            self.exp_map['coverage'] = coverage

    #def names(self,partial_index):
    #    self.super(Explanation, self).names(partial_index)
    


def test_explanation():
    e = Explanation(features=[1,2],names=['test1','test2'],precision=[0.67,0.9],coverage=[0.2,0.05])
