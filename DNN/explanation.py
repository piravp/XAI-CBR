# explanation object, that will be stored in each case
from Induction.Anchor import anchor_explanation
import random
import json
import copy
class Explanation(anchor_explanation.AnchorExplanation, json.JSONEncoder): # extend AnchorExplanation
    def __init__(self,names,feature,precision, prediction,coverage, instance=None, user_defined=True, **args):
        self.user_defined = user_defined # whether or not a particular explanation was made by anchor or not
        if(not all([names,feature,precision,coverage])): # if ether of these have a value other than a list or 0.
            raise ValueError("Need input: names, feature, precision, coverage")
        elif(prediction is None):
            raise ValueError("Need a prediction for explanation")
        else:
            exp_map = {'feature': [], 'mean': [], 'precision': [],
                'coverage': [], 'examples': [], 'prediction':-1} # Initial mapping
            
            # Check if values are correct length with eachother
            if(len(names) != len(feature)):
                #raise ValueError("Length of the input 'names' {names} are not the same as 'feature' {feature} ",names,feature)
                msg = "Length of the input 'names' {}({}) are not the same as 'feature' {}({})".format(
                    names,len(names),feature,len(feature))
                raise ValueError(msg)
            elif (len(coverage) != len(precision)):
                msg = "Length of the input 'coverage' {}({}) are not the same as 'precision' {}({})".format(
                    coverage,len(coverage),precision,len(precision))
                raise ValueError("Length of the input 'coverage' are not the same as 'precision'")
            # TODO: change names list to values from instance
            if(instance is not None): #  we need to swap names list by these numbers instead.
                if(not isinstance(instance,list)):
                    #names = value # this is the one we ultimatly want to store.
                    instance = instance.flatten() # np.array (x,1)-> (x,)
                names = [int(instance[f]) for f in feature] # encoded feature values.
                #Assumes that when instance is not none, then the explanation is from Anchor
                self.user_defined = False
            # otherwise assume that the names are a list of ints
            elif(any([isinstance(n,str)for n in names])): # if any of the names are a string, we need to fix these too.
                raise ValueError("Need an instance to correct string names:{} to encoded values",names)
            # Init input
            exp_map['names'     ] = names
            exp_map['feature'   ] = feature
            exp_map['precision' ] = precision
            exp_map['coverage'  ] = coverage
            exp_map['prediction'] = int(prediction)
            super(Explanation, self).__init__(type_='tabular', exp_map=exp_map, as_html=None)

    def get_explanation(self,decoder_f, decoder_v, partial_index=None):
        # return string: "a"="1" AND "b"="2"
        feature = self.features(partial_index)
        names = self.names(partial_index)
        tmp = []
        for i,f in enumerate(feature): # f is the feature index, name[i] value index
            tmp.append("{} = {}".format(decoder_f[f], decoder_v[f][names[i]]))
        return ' AND '.join(tmp)
    
    def get_explanation_encoded(self):
        # first we need to join feature = name,
        # 
        tmp = []
        names = self.names()
        for i,f in enumerate(self.features()):
            tmp.append("{} = {}".format(f, names[i]))
        return ' AND '.join(tmp)
    
    def get_partial(self,p:int): #return explanation object.
        if(p > len(self.features())):
            p = len(self.features())
        
        exp_copy = copy.deepcopy(self)
        # create explanation object from this one.
        exp_copy.exp_map['names'    ] = self.names(p)
        exp_copy.exp_map['feature'  ] = self.features(p)
        exp_copy.exp_map['precision'] = self.precision(p)
        exp_copy.exp_map['coverage' ] = self.coverage(p)
        return exp_copy # copy part of the explanation, and use it to explain a particular new instance

    def __str__(self): 
        return str(self.exp_map)

    def check_similarity(self,other,p=None): # p: partial_index
        """ 
            Check if two explanations fit on eachother
            fully if partial index not set, assumes p is not too large for both
        """
        # print('from check_similarity() in explanation.py:')
        # print(self.features(), other.features())
        # print(self.names(), other.names())

        if( self.features(p) == other.features(p) and
            self.names(p)    == other.names(p)):
            return True
        return False

    
    def __eq__(self, other):
        """Override the default Equals behavior"""
        return (self.exp_map["names"     ] == other.exp_map["names"    ] and 
                self.exp_map['feature'   ] == other.exp_map['feature'  ] and
                self.exp_map['precision' ] == other.exp_map['precision'] and 
                self.exp_map['coverage'  ] == other.exp_map['coverage' ] and 
                self.exp_map['prediction'] == other.exp_map['prediction'])

    # pylint: disable=E0202
    def default(self, obj): # no need to store whole dictionary.
        return {"__class__":obj.__class__.__name__,
                "names":self.exp_map["names"],
                "feature":self.exp_map['feature'],
                "precision":self.exp_map['precision'],
                "coverage":self.exp_map['coverage'],
                "prediction":self.exp_map['prediction'],
                "user_defined":self.user_defined
                }


def test_explanation():
    import json
    e = Explanation(feature=[1,2],names=['test1','test2'],precision=[0.67,0.9],coverage=[0.2,0.05])

    print(e.features())
    print(e.names())
    print(e.coverage())
    print(e.precision())

    ef = Explanation(feature=[3,5],names=['test1','test2'],precision=[0.67,0.9],coverage=[0.2,0.05],prediction=0)

    # Partial anchor explanation
    print(e.features(0))
    print(e.names(0))
    print(e.coverage(0))
    print(e.precision(0))
    explanation = {1:e.exp_map}
    print(e.exp_map)

    print(json.dumps(explanation))

#test_explanation()