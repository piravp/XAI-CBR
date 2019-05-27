#######################
# Code from repository https://github.com/marcotcr/anchor/tree/master/anchor
# Commented to fit our needs
import anchor_base
import anchor_explanation
import utils

# pip install lime
import lime
import lime.lime_tabular

import collections
import sklearn

import numpy as np
import os
import copy
import string
from io import open
import json

def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

class AnchorTabularExplainer(object):
    """
    Args:
        class_names: list of strings
        feature_names: list of strings
        data: used to build one hot encoder
        categorical_names: map from integer to list of strings, names for each
            value of the categorical features. Every feature that is not in
            this map will be considered as ordinal, and thus discretized.
        ordinal_features: list of integers, features that were discretisized
    """
    def __init__(self, class_names, feature_names, data=None,
                categorical_names=None, ordinal_features=[]):
        self.encoder = collections.namedtuple('random_name',
                                            ['transform'])(lambda x: x)
        self.disc = collections.namedtuple('random_name2',
                                            ['discretize'])(lambda x: x)
        self.categorical_features = [] # name of the different categories
        if categorical_names: 
            # TODO: Check if this n_values is correct!!
            # sort dictionary keys, to match with feature position.
            cat_names = sorted(categorical_names.keys())
            n_values = [len(categorical_names[i]) for i in cat_names]
            #print(feature_names, n_values,sum(n_values))
            #Replace OneHotEncoder with ColumnTransformer, 
            # and transform categorical features with OneHotEncoder
            if(True):
                from sklearn.compose import ColumnTransformer
                categorical_transformer = sklearn.preprocessing.OneHotEncoder(categories="auto")
                # Encode all categorical features using a OneHotEncoder. 
                self.encoder = ColumnTransformer(
                    [("cat", categorical_transformer, cat_names)])
            else:
                self.encoder = sklearn.preprocessing.OneHotEncoder(categories=n_values)
                #self.encoder = sklearn.preprocessing.OneHotEncoder(
                #    categorical_features=cat_names,
                #    categories=n_values)
            self.encoder.fit(data) # Fit one_hot_encoder to train_data
            self.categorical_features = cat_names#self.encoder.categorical_features
            #self.categorical_features = self.encoder.ColumnTransformer
        if len(ordinal_features) == 0: # If no list of features that are continous. 
            self.ordinal_features = [ # get all features that are not categorical.
                x for x in range(len(feature_names)) if x not in self.categorical_features]

        # Init vars
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_names = categorical_names

    def fit(self, train_data, train_labels, validation_data,
            validation_labels, discretizer='quartile'):
        """
        Fit the anchor_tabular object to the dataset, and discretize as needed.
        Args:
            train_data: numpy 2d array
            train_labels: labels for training data. May be
                used by discretizer.
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer:  Options are 'quartile', 'decile', 'entropy' or a BaseDiscretizer
                instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.min = {} # min value
        self.max = {} # max value
        self.std = {} # standard deviation
        # init variables
        self.train = train_data
        self.train_labels = train_labels
        self.validation = validation_data
        self.validation_labels = validation_labels
        # Init StandarScalar, Standardize features by removing the mean and scaling to unit variance
        self.scaler = sklearn.preprocessing.StandardScaler()
        # Compute the mean and std to be used for later scaling.
        self.scaler.fit(train_data) 
        # Discretisize continous features with coresponding discretisizer.
        if discretizer == 'quartile': 
            self.disc = lime.lime_tabular.QuartileDiscretizer(train_data,
                                                        self.categorical_features,
                                                        self.feature_names)
        elif discretizer == 'decile':
            self.disc = lime.lime_tabular.DecileDiscretizer(train_data,
                                                    self.categorical_features,
                                                    self.feature_names)
        # Discretize with entropy, instead of percentiles. This will split on information gain, against the labels.
        elif discretizer == 'entropy':
            self.disc = lime.lime_tabular.EntropyDiscretizer(train_data,
                                                    self.categorical_features,
                                                    self.feature_names,
                                                    labels=train_labels)
        else:
            raise ValueError('Discretizer must be quartile, decile or entropy')
        # Discretizise training and validation datasets, if needed.
        self.d_train = self.disc.discretize(self.train)
        self.d_validation = self.disc.discretize(self.validation)



        self.categorical_names.update(self.disc.names) # Add discretized feature mapping of contious features.
        # Ordinal_features is every feature that is not categorical. (that have been discretized)
        self.ordinal_features = [x for x in range(self.d_validation.shape[1])
                            if x not in self.categorical_features]
        self.categorical_features += self.ordinal_features

        for f in range(train_data.shape[1]): # for each feature. (column)
            # if feature is categorical and not ordinal feature.
            if f in self.categorical_features and f not in self.ordinal_features:
                continue
            self.min[f] = np.min(train_data[:, f])
            self.max[f] = np.max(train_data[:, f])
            self.std[f] = np.std(train_data[:, f])
        
        # Print values, to check what we got.

    def sample_from_train(self, conditions_eq, conditions_neq, conditions_geq,
                        conditions_leq, num_samples, validation=True):
        """
        Select sample from training set, to be used to train?
        """
        # set training_set to self.train if not validations set present.
        train = self.train if not validation else self.validation
        # set discretized training set to training if not discretized validation set present.
        d_train = self.d_train if not validation else self.d_validation
        # Index of which training exaples to choose (rows), 
        idx = np.random.choice(range(train.shape[0]), num_samples,
                            replace=True) # select n random unique index values in dataset.
        sample = train[idx] # Select sample from dataset
        d_sample = d_train[idx] # Select corresponding samples discretized
        for f in conditions_eq: # 
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)
            
        for f in conditions_geq:
            idx = d_sample[:, f] <= conditions_geq[f]
            if f in conditions_leq:
                idx = (idx + (d_sample[:, f] > conditions_leq[f])).astype(bool)
            if idx.sum() == 0:
                continue
            options = d_train[:, f] > conditions_geq[f]
            if f in conditions_leq:
                options = options * (d_train[:, f] <= conditions_leq[f])
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:
                to_rep = np.random.choice(train[options, f], idx.sum(),
                                        replace=True)
            sample[idx, f] = to_rep
        for f in conditions_leq:
            if f in conditions_geq:
                continue
            idx = d_sample[:, f] > conditions_leq[f]
            if idx.sum() == 0:
                continue
            options = d_train[:, f] <= conditions_leq[f]
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:
                to_rep = np.random.choice(train[options, f], idx.sum(),
                                        replace=True)
            sample[idx, f] = to_rep
        return sample

    def transform_to_examples(self, examples, features_in_anchor=[],
                            predicted_label=None):
        ret_obj = []
        if len(examples) == 0:
            return ret_obj
        weights = [int(predicted_label) if x in features_in_anchor else -1
                for x in range(examples.shape[1])]
        examples = self.disc.discretize(examples)
        for ex in examples:
            values = [self.categorical_names[i][int(ex[i])]
                    if i in self.categorical_features
                    else ex[i] for i in range(ex.shape[0])]
            ret_obj.append(list(zip(self.feature_names, values, weights)))
        return ret_obj

    def to_explanation_map(self, exp):
        def jsonize(x): return json.dumps(x)
        instance = exp['instance']
        predicted_label = exp['prediction']
        predict_proba = np.zeros(len(self.class_names))
        predict_proba[predicted_label] = 1

        examples_obj = []
        for i, temp in enumerate(exp['examples'], start=1):
            features_in_anchor = set(exp['feature'][:i])
            ret = {}
            ret['coveredFalse'] = self.transform_to_examples(
                temp['covered_false'], features_in_anchor, predicted_label)
            ret['coveredTrue'] = self.transform_to_examples(
                temp['covered_true'], features_in_anchor, predicted_label)
            ret['uncoveredTrue'] = self.transform_to_examples(
                temp['uncovered_true'], features_in_anchor, predicted_label)
            ret['uncoveredFalse'] = self.transform_to_examples(
                temp['uncovered_false'], features_in_anchor, predicted_label)
            ret['covered'] =self.transform_to_examples(
                temp['covered'], features_in_anchor, predicted_label)
            examples_obj.append(ret)

        explanation = {'names': exp['names'],
                    'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                    'supports': exp['coverage'],
                    'allPrecision': exp['all_precision'],
                    'examples': examples_obj,
                    'onlyShowActive': False}
        weights = [-1 for x in range(instance.shape[0])]
        instance = self.disc.discretize(exp['instance'].reshape(1, -1))[0]
        values = [self.categorical_names[i][int(instance[i])]
                if i in self.categorical_features
                else instance[i] for i in range(instance.shape[0])]
        raw_data = list(zip(self.feature_names, values, weights))
        ret = {
            'explanation': explanation,
            'rawData': raw_data,
            'predictProba': list(predict_proba),
            'labelNames': list(map(str, self.class_names)),
            'rawDataType': 'tabular',
            'explanationType': 'anchor',
            'trueClass': False
        }
        return ret

    def as_html(self, exp, **kwargs):
        """bla"""
        exp_map = self.to_explanation_map(exp)

        def jsonize(x): return json.dumps(x)
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()
        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "tabular", {explanation_type});
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(exp_map['labelNames']),
                            predict_proba=jsonize(exp_map['predictProba']),
                            true_class=jsonize(exp_map['trueClass']),
                            explanation=jsonize(exp_map['explanation']),
                            raw_data=jsonize(exp_map['rawData']),
                            explanation_type=jsonize(exp_map['explanationType']))
        out += u'</body></html>'
        return out

    def get_sample_fn(self, data_row, classifier_fn, desired_label=None):
        def predict_fn(x): # define function, to predict input predict(encode(input))
            return classifier_fn(self.encoder.transform(x))
        true_label = desired_label # 
        if true_label is None: # If we don't know the labels of the input, do a prediction.
            true_label = predict_fn(data_row.reshape(1, -1))[0]
        # must map present here to include categorical features (for conditions_eq), and numerical features for geq and leq
        mapping = {}
        data_row = self.disc.discretize(data_row.reshape(1, -1))[0]
        for f in self.categorical_features: # for every feature that is categorical
            if f in self.ordinal_features: # if the feature is a bin.
                for v in range(len(self.categorical_names[f])):
                    idx = len(mapping)
                    if data_row[f] <= v and v != len(self.categorical_names[f]) - 1:
                        mapping[idx] = (f, 'leq', v) # less than or equal to
                        # names[idx] = '%s <= %s' % (self.feature_names[f], v)
                    elif data_row[f] > v:
                        mapping[idx] = (f, 'geq', v) # greater than or equal to
                        # names[idx] = '%s > %s' % (self.feature_names[f], v)
            else: # if the feature is categorical
                idx = len(mapping)
                mapping[idx] = (f, 'eq', data_row[f])
            # names[idx] = '%s = %s' % (
            #     self.feature_names[f],
            #     self.categorical_names[f][int(data_row[f])])

        def sample_fn(present, num_samples, compute_labels=True, validation=True):
            conditions_eq = {}
            conditions_leq = {}
            conditions_geq = {}
            for x in present:
                f, op, v = mapping[x]
                if op == 'eq':
                    conditions_eq[f] = v
                if op == 'leq':
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    conditions_leq[f] = min(conditions_leq[f], v)
                if op == 'geq':
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)
            # conditions_eq = dict([(x, data_row[x]) for x in present])
            raw_data = self.sample_from_train(
                conditions_eq, {}, conditions_geq, conditions_leq, num_samples,
                validation=validation)
            d_raw_data = self.disc.discretize(raw_data)
            data = np.zeros((num_samples, len(mapping)), int)
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (d_raw_data[:, f] == data_row[f]).astype(int)
                if op == 'leq':
                    data[:, i] = (d_raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (d_raw_data[:, f] > v).astype(int)
            # data = (raw_data == data_row).astype(int)
            labels = []
            if compute_labels:
                labels = (predict_fn(raw_data) == true_label).astype(int)
            return raw_data, data, labels
        return sample_fn, mapping

    def explain_instance(self, data_row, classifier_fn, threshold=0.95,
                        delta=0.1, tau=0.15, batch_size=100,
                        max_anchor_size=None,
                        desired_label=None,
                          beam_size=4, **kwargs):
        # It's possible to pass in max_anchor_size
        sample_fn, mapping = self.get_sample_fn(
            data_row, classifier_fn, desired_label=desired_label)
        # return sample_fn, mapping
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, max_anchor_size=max_anchor_size,
            **kwargs)
        self.add_names_to_exp(data_row, exp, mapping)
        exp['instance'] = data_row
        # Store prediction from network, on dataset
        exp['prediction'] = classifier_fn(self.encoder.transform(data_row.reshape(1, -1)))[0]

        explanation = anchor_explanation.AnchorExplanation('tabular', exp, self.as_html)
        return explanation

    def add_names_to_exp(self, data_row, hoeffding_exp, mapping):
        # TODO: precision recall is all wrong, coverage functions wont work
        # anymore due to ranges
        idxs = hoeffding_exp['feature']
        hoeffding_exp['names'] = []
        hoeffding_exp['feature'] = [mapping[idx][0] for idx in idxs]
        ordinal_ranges = {}
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'geq' or op == 'leq':
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float('-inf'), float('inf')]
            if op == 'geq':
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == 'leq':
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)

        handled = set()
        for idx in idxs:
            f, op, v = mapping[idx]
            # v = data_row[f]
            if op == 'eq':
                fname = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v]
                            or '>' in self.categorical_names[f][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.categorical_names[f][v])
                else:
                    fname = '%s%.2f' % (fname, v)
            else:
                if f in handled:
                    continue
                geq, leq = ordinal_ranges[f]
                fname = ''
                geq_val = ''
                leq_val = ''
                if geq > float('-inf'):
                    if geq == len(self.categorical_names[f]) - 1:
                        geq = geq - 1
                    name = self.categorical_names[f][geq + 1]
                    if '<' in name:
                        geq_val = name.split()[0]
                    elif '>' in name:
                        geq_val = name.split()[-1]
                if leq < float('inf'):
                    name = self.categorical_names[f][leq]
                    if leq == 0:
                        leq_val = name.split()[-1]
                    elif '<' in name:
                        leq_val = name.split()[-1]
                if leq_val and geq_val:
                    fname = '%s < %s <= %s' % (geq_val, self.feature_names[f],
                                            leq_val)
                elif leq_val:
                    fname = '%s <= %s' % (self.feature_names[f], leq_val)
                elif geq_val:
                    fname = '%s > %s' % (self.feature_names[f], geq_val)
                handled.add(f)
            hoeffding_exp['names'].append(fname)