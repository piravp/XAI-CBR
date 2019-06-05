""" 
    This is the main file used for the experimentation.
    By seperating the experiments and seeding the randomization at each step, it will be reproduceable.

    Its important that the myCBR rest projet is running.
    # COMMAND TO START PROGRAM (in CBR/libs/mycbr-rest)
    > java -DMYCBR.PROJECT.FILE=<absolute_path>\XAI-CBR\CBR\projects\adult\adult.prj -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar
    
    ONLY WORKS ON WINDOWS

    For more Info see README file.
    
    # Better Comments extension (VS code) used for easier reading of different comments.
"""
# Turn off warnings
import os
import signal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# Helpers libs
import pathlib
import numpy as np
import pandas as pd
import subprocess
from subprocess import Popen,CREATE_NEW_CONSOLE,PIPE
import time
import json
import argparse

# Main libs
from DNN.kera import network # Import black-box (ANN)
from DNN.kera import pre_processing # Import dataset preprocessing
from DNN.Induction.Anchor import anchor_tabular, utils # explanatory framework
from DNN import knowledge_base,explanation
from CBR.src import CBRInterface
from CBR.src.case import Case
# Integrated gradients
from deepexplain.tensorflow import DeepExplain
from keras import backend as K
from keras.models import Model

from collections import defaultdict

class Experiments():
    def __init__(self, verbose=False):
        #* Init Dataset
        self.dataman = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
        self.dataset = self.dataman.ret
        if(verbose):
            print("Keys", self.dataset.__dict__.keys())
        
        #* Init Anchors
        explainer = anchor_tabular.AnchorTabularExplainer(
        self.dataset.class_names, self.dataset.feature_names,
        self.dataset.data_train, self.dataset.categorical_names)
        
        #! Explainer.encoder.transform return sparse matrix, instead of dense np.array
        explainer.fit(self.dataset.data_train, self.dataset.train_labels, 
                    self.dataset.data_validation, self.dataset.validation_labels) # Fit the labels to the explainer.
        self.anchors_explainer = explainer

        #* Init integrated gradients
        cat_names = sorted(self.dataset.categorical_names.keys())
        self.n_values = [len(self.dataset.categorical_names[i]) for i in cat_names]

        #* Init BlackBox
        self.bb = network.BlackBox(name="NN-adult-5",c_path="NN-Adult-5/NN-Adult-5-8531.hdf5")
        if(verbose):
            self.bb.evaluate(data_train=self.anchors_explainer.encoder.transform(self.dataset.data_train).toarray(),train_labels=self.dataset.train_labels,
                    data_test=self.anchors_explainer.encoder.transform(self.dataset.data_test).toarray(),test_labels=self.dataset.test_labels)

        # Init CaseBase
        self.CBR = CBRInterface.RESTApi() # load CBR restAPI class, for easy access.

    # ----------------------------------------------------------------------------------------------------- #
    #                              H A N D L I N G   M Y C B R                                              #
    #                       Starting, stopping and checking status of myCBR                                 #
    # ----------------------------------------------------------------------------------------------------- #

    def start_MyCBR(self, project, jar, storage=False): # Start myCBR project file # TO put everything into the same console, remove flag.
        print("Starting myCBR Rest server") # ,stdout=PIPE,stderr=PIPE
        self.process = Popen(["java","-DMYCBR.PROJECT.FILE={}".format(project),
                            "-Dsave={}".format(storage),"-jar",str(jar)],shell=True)#, creationflags=CREATE_NEW_CONSOLE)
        # Return once it is up and running

    def myCBR_running(self):
        status = self.CBR.checkStatus()
        count = 0
        while(status == 404 or status == 500): # if not sucessfull, keep sleeping
            count += 1
            time.sleep(5)
            status = self.CBR.checkStatus()
            if(count == 8):
                print("Took too long time to start up project")
                self.stop_MyCBR()
                exit()
            print("Current status code myCBR:",status)
        print("------------MyCBR ready------------")
        # TODO: count number of steps required

    def stop_MyCBR(self):
        if(self.process.poll() is None): # check if process is still running.
            print("KILLING RUNNING PROCESSES BEFORE EXITING")
            if os.name == 'nt': # teriminate doest work on windows.
                subprocess.call(['taskkill','/F','/T','/PID',str(self.process.pid)])
            self.process.terminate() # terminalte process

    def start_server(self, project, jar, storage=False):
        # Simply start the Server, and dont stop running
        subprocess.call(["java","-DMYCBR.PROJECT.FILE={}".format(project.absolute()),
                            "-Dsave={}".format(storage),"-jar",str(jar.absolute())])
        #self.start_MyCBR(project=project,jar=jar)
        self.myCBR_running() # Continue running.

        conceptID = self.CBR.getConceptID()
        casebaseID = self.CBR.getCaseBaseID()
        print(conceptID, casebaseID)

    # ----------------------------------------------------------------------------------------------------- #
    #                               G R A D I E N T  A N D  A N C H O R                                     #
    #           Helpers to generate integrated gradient (single and multiple) and anchor explanations       #
    # ----------------------------------------------------------------------------------------------------- #

    def get_attribution(self, instance):
        # Get attribution from an instance on the black box (ANN)
        #attribution_weights_full = []
        with DeepExplain(session=K.get_session()) as de: # Init session, to keep gradients.
            input_tensors = self.bb.model.inputs
            output_layer = self.bb.model.outputs
            fModel = Model(inputs=input_tensors, outputs=output_layer)
            target_tensor = fModel(input_tensors)

            # TODO: Handle multiple instances at once, to save processing time.

            attribution_weights_instance = []
            instance = instance.reshape(1,-1)
            attribution = de.explain('intgrad',target_tensor, input_tensors, 
                                    self.anchors_explainer.encoder.transform(instance).toarray())
            one_hot_vector = attribution[0][0]

            # Compress one_hot_encoded input vector attribution into one per original attribute. (71 -> 12) 
            start = 0
            for n in self.n_values: # itterate over each categorie
                compressed = sum(one_hot_vector[start:start+n])
                attribution_weights_instance.append(np.around(compressed, decimals=4))
                start += n # increase start slice

        return attribution_weights_instance # return list of attributions.

    def get_attribution_multiple(self, instances, compress=True):
        attribution_weights_full = []
        with DeepExplain(session=K.get_session()) as de:
            input_tensors = self.bb.model.inputs
            output_layer = self.bb.model.outputs
            fModel = Model(inputs=input_tensors, outputs=output_layer)
            target_tensor = fModel(input_tensors)
            attribution = de.explain('intgrad',target_tensor, input_tensors, 
                                        [self.anchors_explainer.encoder.transform(instances).toarray()])
            for attrib in attribution[0]:
                if(not compress): # If we to compress or not.
                    attribution_weights_full.append(attrib.tolist())
                else:
                    attribution_weights_instance = []
                    # Compress attribution vector (71 elements, based on one-hot-vector) to only needing 12 elements
                    start = 0
                    for n in self.n_values:
                        compressed = sum(attrib[start:start+n])
                        attribution_weights_instance.append(np.around(compressed, decimals=4))
                        start += n                                                          # increase start slice
                    attribution_weights_full.append(attribution_weights_instance)      # need to be converted to string to save in case-base
                
        return attribution_weights_full

    def get_explanation_prediction(self, encoding):
        explanations = []
        predictions = []
        for i, instance in enumerate(encoding):
            exp = self.anchors_explainer.explain_instance(instance, self.bb.predict, threshold=0.95,verbose=False)
            custom_exp = explanation.Explanation(**exp.exp_map) # create explanation object,
            explanations.append(custom_exp)
            predictions.append(custom_exp.exp_map['prediction'])
        return explanations, predictions # return two lists, one with explanation and one with predictions. 


    # ----------------------------------------------------------------------------------------------------- #
    #                                 H A N D L I N G  C A S E S                                            #
    #          Generate cases from validation set, test set and divide cases in batches                     #
    # ----------------------------------------------------------------------------------------------------- #

    # Used to divide list l of cases into batches of n
    def divide_batches(self, l, n): 
        # looping till length l 
        for i in range(0, len(l), n):  
            yield l[i:i + n] 

    def get_cases(self, instances, encoding, KB, compress): 
        attributions = self.get_attribution_multiple(encoding, compress)
        explanations, predictions = self.get_explanation_prediction(encoding)

        case_objects = [] # list of cases
        for i, inc in enumerate(instances):  
            exp_id = KB.add_knowledge(explanations[i])
            #TODO: clean up the case_generation (not a very good approach ..)
            case_objects.append(Case(age=inc[0], workclass=inc[1], education=inc[2], martial_status=inc[3], occupation=inc[4],
                relationship=inc[5], race=inc[6], sex=inc[7], capital_gain=inc[8], capital_loss=inc[9],
                hours_per_week=inc[10],country=inc[11],
                weight=attributions[i], prediction = predictions[i], explanation = exp_id, KB=KB))
        return case_objects

    # def generate_init_cases(self,N, compress=True):
    #     """ 
    #         N is number of cases to generate
    #         Generate initial cases from validation data to query the CBR system with.
    #         Returns a list of case objects.
    #     """
    #     # Select random number of indexes from validation indexes
    #     idx_cases_val = np.random.choice(self.dataset.validation_idx, N, replace=False)# non repeating instances

    #     # Select cases from indexes, on the dataset before splitting, in readable form and encoded for black-box input.
    #     init_cases = self.dataset.data_test_full.values[idx_cases_val]
    #     init_cases_enc = self.dataset.data_test_enc_full[idx_cases_val]

    #     cases = self.get_cases(instances = init_cases, encoding=init_cases_enc, KB = self.KB, compress=compress)
    #     return cases 

    # def generate_test_cases(self, N, compress=True):
    #     """ 
    #         N is number of cases to generate
    #         Generate initial knowledge cases from test data to be put into the CBR system.
    #         Returns a list of case objects.
    #     """
    #     # select random number of index from test indexes, to vertify the similarity.
    #     idx_cases_test = np.random.choice(self.dataset.test_idx, N, replace=False)# non repeating instances

    #     test_cases = self.dataset.data_test_full.values[idx_cases_test]
    #     test_cases_enc = self.dataset.data_test_enc_full[idx_cases_test]

    #     cases_test = self.get_cases(instances = test_cases, encoding=test_cases_enc, KB = self.KB_test, compress=compress)
    #     return cases_test

    def generate_cases(self, N, N_T, unique=False, compress=True):
        # Genererate case objects from these.   
        """ 
            N is number of cases to generate
            Generate initial cases from validation data to query the CBR system with.
            Returns a list of case objects.
        """
        # Select random number of indexes from validation indexes
        idx_cases_val = np.random.choice(self.dataset.validation_idx, N, replace=False)# non repeating instances

        # Select cases from indexes, on the dataset before splitting, in readable form and encoded for black-box input.
        init_cases = self.dataset.data_test_full.values[idx_cases_val]
        init_cases_enc = self.dataset.data_test_enc_full[idx_cases_val]

        idx_cases_test = np.random.choice(self.dataset.test_idx, N_T*2, replace=False)# non repeating instances

        test_cases = self.dataset.data_test_full.values[idx_cases_test]
        test_cases_enc = self.dataset.data_test_enc_full[idx_cases_test]

        # Whether or not we want unique cases in the test_set, we don't bother with the initial cases. 
        if(unique):
            # Check if cases_test are unique from init_cases
            indexes = []
            for i, t_case in enumerate(test_cases_enc):
                if(check_contains(t_case, init_cases_enc)):
                    indexes.append(i)

            for i in indexes: # Set elements to None, for easy of deletion.
                test_cases[i] = np.nan
                test_cases_enc[i] = np.nan

            # Remove these None elements. 
            test_cases =     [case for case in test_cases if not pd.isnull(case).any()]
            test_cases_enc = [case for case in test_cases_enc if not pd.isnull(case).any()]
            if(len(test_cases) != len(test_cases_enc)):
                raise ValueError("Not equal lenghts between encoded and raw test cases")

        cases = self.get_cases(instances = init_cases, encoding=init_cases_enc, KB = self.KB, compress=compress)
        cases_test = self.get_cases(instances = test_cases[:N_T], encoding=test_cases_enc[:N_T], KB = self.KB_test, compress=compress)

        return cases, cases_test

    def run_test(self):
        print(self.dataset.__dict__.keys())
        # simply test some different things.
        np.random.seed(1) # init seed
        # Say we want to select X number of instances and put into the CaseBase.
        # Get cases from validation dataset, and prediction from

        self.dataset.data_validation # Attributes, encoded and all
        self.dataset.validation_labels # labels, 0s and 1s
        
        n = 10

        # Select random number of indexes from validation indexes
        idx_cases_val = np.random.choice(self.dataset.validation_idx,n,replace=False)# non repeating instances
        idx_cases_val = np.sort(idx_cases_val) # easier to work with

        # select random number of index from test indexes, to vertify the similarity.
        idx_cases_test = np.random.choice(self.dataset.test_idx,n,replace=False)# non repeating instances
        idx_cases_test = np.sort(idx_cases_test) # easier to work with

        # Select cases from indexes, on the dataset before splitting, in readable form and encoded for black-box input.
        init_cases = self.dataset.data_test_full.values[idx_cases_val]
        init_cases_enc = self.dataset.data_test_enc_full[idx_cases_val]
        init_cases_labels = self.dataset.labels_test[idx_cases_val] # labels corresponding to input.

        test_cases = self.dataset.data_test_full.values[idx_cases_test]
        test_cases_enc = self.anchors_explainer.encoder.transform(self.dataset.data_test_enc_full[idx_cases_test])

        # Now we can generate cases from these lists, and vertify their results in the next examples.

        # Generate cases from the list, with or without explanation parts.

        start = time.clock()
        attributions = self.get_attribution_multiple(init_cases_enc)
        end = time.clock()
        print("Seconds used to generate attribution weights:", end-start)

        start = time.clock()
        explanations = []
        predictions = []
        # Generate explanations for each case.
        for i, instance in enumerate(init_cases_enc):
            exp = self.anchors_explainer.explain_instance(instance, self.bb.predict, threshold=0.95,verbose=False)
            custom_exp = explanation.Explanation(**exp.exp_map)
            explanations.append(custom_exp)
            predictions.append(custom_exp.exp_map['prediction']) 
            print("Generated explanation for case {}.".format(i))
        end = time.clock()
        print("Seconds used to generate anchor explanations:", end-start)

        # Create cases from these

        initial_case_objects = [] # list of cases

        for i, inc in enumerate(init_cases):  
            initial_case_objects.append(Case(age=inc[0], workclass=inc[1], education=inc[2], martial_status=inc[3], occupation=inc[4],
                relationship=inc[5], race=inc[6], sex=inc[7], capital_gain=inc[8], capital_loss=inc[9],
                hours_per_week=inc[10],country=inc[11],
                weight=str(attributions[i]), prediction = predictions[i], explanation = i))

        print(json.dumps(initial_case_objects[0], default=Case.default))
        print(json.dumps(initial_case_objects, default=Case.default))
        #print(json.dumps(initial_case_objects, default=Case.default))
        conceptID = self.CBR.getConceptID()
        casebaseID = self.CBR.getCaseBaseID()
        
        print(conceptID, casebaseID)
        print(self.CBR.getAlgamationFunctions(conceptID = conceptID))
        
        # We need to get representations that can be used.
        
        # generate cases from these
        # We need to get the prediction from test_cases.
        
    # ----------------------------------------------------------------------------------------------------- #
    #                               E  X  P  E  R  I  M  E  N  T  S                                         #
    #          ************************************************************************                     #
    # ----------------------------------------------------------------------------------------------------- #

    def run_experiment_fill(self, N, project, jar, storage=True):
        """
            Experiment to check whether the case-base is populated correctly (with cases from validation set).
        """
        # Init random seed for consistency
        np.random.seed(1) 

        # Start CBR project
        self.start_MyCBR(project, jar, storage) 
        self.myCBR_running() # Continue running.

        # Get the concept ID, and CaseBase ID
        conceptID = self.CBR.getConceptID()
        casebaseID = self.CBR.getCaseBaseID()
        print('ConceptID:', conceptID, 'CasebaseID:', casebaseID)

        # Check whether or not the casebase is filled with cases
        size = self.CBR.getCaseBaseSize(conceptID = conceptID, casebaseID = casebaseID)
        if(size != 0):
            print('> WARNING! Cases in this test are not meant to be overwritten.')
            raise ValueError("The case-base is not empty")


        # Select random number of indexes from validation indexes
        idx_cases_val = np.random.choice(self.dataset.validation_idx, N, replace=False)# non repeating instances

        # Select cases from indexes, on the dataset before splitting, in readable form and encoded for black-box input. Validation set.
        init_cases = self.dataset.data_test_full.values[idx_cases_val]
        init_cases_enc = self.dataset.data_test_enc_full[idx_cases_val]
       
        #Generate validation cases.
        attributions = self.get_attribution_multiple(init_cases_enc)
        explanations, predictions = self.get_explanation_prediction(init_cases_enc)

        # Create knowledge-base
        print('Checking knowledge-base...')
        self.KB = knowledge_base.KnowledgeBase("exp_fill")   # Init knowledge base for validation data
        self.KB.reset_knowledge() # empty the knowledge-base before we begin.
        self.KB_test = knowledge_base.KnowledgeBase("exp_fill_test")   # Init knowledge base for test data (not used but need to be init)
        self.KB.reset_knowledge() # empty the knowledge-base before we begin.


        # Genererate case objects from these.   
        cases, _ = self.generate_cases(N=N, N_T=1)
        
        # Add all of the cases (from validation set) to the case-base
        # NOTE! HTTP header is limited to about 10 cases, so adding cases need to be done in batches of max 10.
        batch_size = 10
        cases_batch = self.divide_batches(l=cases, n=batch_size)
        for i, batch in enumerate(cases_batch):
            print('Adding batch {} with batch size {}...'.format(i, len(batch)))
            print(self.CBR.addInstancesCases(casebaseID='cb0', conceptID='Person', cases=batch))
            print()


    def run_experiment_sim(self, N_T, project, jar, storage=False):
        """
            Experiment to check if similarity measures are useful
            by comparing cases from test-cases against cases already in cb from validation-set
        """

        # Init same seed every time for consistency
        np.random.seed(1) 

        # Start CBR project
        self.start_MyCBR(project, jar, storage) 
        self.myCBR_running() # Continue running.

        # Get the concept ID, and CaseBase ID
        conceptID = self.CBR.getConceptID()
        casebaseID = self.CBR.getCaseBaseID()
        print('ConceptID:', conceptID, 'CasebaseID:', casebaseID)

        # Check whether or not the casebase is filled with cases
        size = self.CBR.getCaseBaseSize(conceptID = conceptID, casebaseID = casebaseID)
        if(size == 0):
            raise ValueError("Case-base is empty")


        # Init knowledge base for validation data
        print('Checking knowledge-base...')
        self.KB = knowledge_base.KnowledgeBase("exp_sim")
        # self.KB.reset_knowledge() # empty the knowledge-base before we begin.

        self.KB_test = knowledge_base.KnowledgeBase("exp_sim_test")
        self.KB_test.reset_knowledge() # empty the knowledge-base before we begin.


        _dummy, test_cases = self.generate_cases(N=1, N_T=N_T)
        # test_cases = self.generate_test_cases(N_T) 
        print('# of cases from test set:', len(test_cases))
    

        # single_test_case = test_cases[0]

        for i in range(0, N_T):
            print(">>>>>>>>> Adding test case {} <<<<<<<<<<<<<".format(i))
            t_case = test_cases[i]
            self.addTestCaseTemporarily(testCase=t_case, topK=3)
            print('\n\n')
        # Delete after use so that one case is not considered before testing the next case


        # Generate Case objects from test_cases, add to separate knowledge-base

        # self.CBR.retrieve_k_sim_byID(conceptID=conceptID, casebaseID=casebaseID, queryID='', k=5)
        # 1. Retreve cases from CB
        # 2. Perform k most similar and find most similar cases
        # 3. Retrieve explanation for most similar case
        # 4. Present explanation to user

        # print(self.CBR.getAlgamationFunctions(conceptID = conceptID))


    # Note that there are no persistent effects as myCBR is run with save(storage) flag set to false
    def addTestCaseTemporarily(self, testCase, topK):
        caseID = self.CBR.addInstancesCases(casebaseID='cb0', conceptID='Person', cases=[testCase])
        caseID = eval(caseID)[0] #Convert from string to list and get only item
        print('caseID', caseID)

        df = self.CBR.retrieve_k_sim_byID(conceptID='Person', casebaseID='cb0', queryID=caseID, k=topK)
        df = df.iloc[1:]    # Exclude the test-case itself which is also returned as it is part of the cb
        # print(df)


        # Print explanation

        # Explanation for test case
        print('----EXPLANATION FOR TEST CASE:----')
        testExpId = testCase.explanation 
        exp = self.KB_test.get(testExpId)
        # print(exp)
        print(exp.get_explanation(self.dataset.feature_names,self.dataset.categorical_names))

        # # Explanation for val case (in case-base)
        # res = self.CBR.getSingleInstance(conceptID='Person', casebaseID='cb0', instanceID='Person-cb010')
        # valExpId = int(res["case"]["Explanation"])
        # test_exp = self.KB.get(valExpId)
        # print(test_exp.get_explanation(self.dataset.feature_names,self.dataset.categorical_names))
        # print()

        print('----TOP K MOST SIMILAR VALIDATION CASES:----')  
        for casename, row in df.iterrows():
            # Explanation for val case (in case-base)
            print('*', casename, row.to_string())
            res = self.CBR.getSingleInstance(conceptID='Person', casebaseID='cb0', instanceID=casename)
            res = res["case"]
            c = Case( res['Age'], res['CapitalGain'], res['CapitalLoss'], res['Country'], res['Education'], 
                       res['Explanation'], res['HoursPerWeek'], res['MaritalStatus'], res['Occupation'],
                        res['Prediction'], res['Race'], res['Relationship'], res['Sex'], res['Weight'], 
                         res['Workclass'], self.KB)
            
            valExpId = c.explanation #int(res["case"]["Explanation"])
            exp_val = self.KB.get(valExpId)
            print(exp_val.get_explanation(self.dataset.feature_names,self.dataset.categorical_names))
            # Delete case from case-base after getting explanation
            self.CBR.deleteInstance(casebaseID='cb0', conceptID='Person', instanceID=casename)


        #TODO: Case må slettes etter å ha lagt inn
        


    def run_experiment_1(self, N, N_T, M, project, jar, storage=False, unique=True, compress=True): # N is number of cases in casebase, M is number of retrievals
        # Note that there are no persistent effects as myCBR is run with save(storage) flag set to false
        """ 
            ? Test whether or not we are able to use previous explanations in the CBR system 
            
            Pre initiate the case-base with N randomly selected cases from the validation set
            Preform M random retrievals and check if an explanation can be found.

            Count how many explanations are correct, with respect to the explanation generated for the specific instance.
        """
        np.random.seed(1) # init seed
        #Load the case-base system

        #Initiate cases into the project
        self.start_MyCBR(project, jar, storage) # Start CBR project.
        self.myCBR_running() # Continue running.

        # Get the concept ID, and CaseBase ID
        conceptID = self.CBR.getConceptID()
        casebaseID = self.CBR.getCaseBaseID()
        print(conceptID, casebaseID)
        
        # Check whether or not the casebase if filled with cases or not.
        size = self.CBR.getCaseBaseSize(conceptID = conceptID, casebaseID = casebaseID)
        if(size != 0):
            raise ValueError("The case-base is not empty")

        # Init knowledge base for validation data
        self.KB = knowledge_base.KnowledgeBase("exp1")
        self.KB.reset_knowledge() # empty the knowledge-base before we begin.

        self.KB_test = knowledge_base.KnowledgeBase("exp1_test")
        self.KB_test.reset_knowledge() # empty the knowledge-base before we begin.

        init_cases, test_cases = self.generate_cases(N, N_T, unique=unique,compress=compress)
        
        print(len(init_cases), len(test_cases))


        # * PERFORM EXPERIMENT
        for n in range(M,N+M,M):
            print("Adding cases from",n-M,"to",n)
            # * Add cases to CaseBase in bulks.
            print("Cases added:",self.CBR.addInstancesCases(casebaseID = casebaseID, conceptID = conceptID, cases=init_cases[n-M:n]))

        # INIT EXPERIMENT:
    
        # Randomly select N from validation set.
        
        # Randomly select M from test set to check against.
        
    def run_experiment_2(self,project, jar, storage=False):
        """  
            ? Test wheter or not we are able to use previous explanations in tandem with custom explanations given by a domain expert.

            Pre Initiate the knowledge-base with custom explanation anchors, to explain a given case-instance. 
            If the expert knowledge fit a new problem, then it is used instead of from the case-base. 

        """
        np.random.seed(1) # init seed

    def run_experiment_3(self,project, jar, storage=False):
        """
            ? Test whether the attribution score from integradet gradients can be used to help with the retrieval of relevant cases 

        """
        np.random.seed(1) # init seed

    def run_experiment_4(self,N,N_T,M, unique=True, compress=True):
        """
            ? Test whether the attribution score can be used for retrieval alone on the CaseBase

        """
        ############# initializing #############
        np.random.seed(1) # init seed
        #Load the case-base system

        # Init knowledge base for validation data
        self.KB = knowledge_base.KnowledgeBase("exp4")
        self.KB.reset_knowledge() # empty the knowledge-base before we begin.

        self.KB_test = knowledge_base.KnowledgeBase("exp4_test")
        self.KB_test.reset_knowledge() # empty the knowledge-base before we begin.

        # Genererate case objects from these.   
        # N, N_T, unique=False, compress=True
        init_cases, test_cases = self.generate_cases(N, N_T, unique=unique,compress=compress)
        ################### STARTING EXPERIMENT ###################

        print("Generating",N,"cases and",N_T, " test cases completed")
        print("Cosine Similarity, self:",test_cases[0].checkCosineDistance(test_cases[0]))
        print("Eucluidian similarity, self:", test_cases[0].checkEuclidianDistance(test_cases[0]))

        def experiment_4(cases, test_case, query, measurements_dict, n):
            for k, (sim, i) in enumerate(query): # query results, (similarity, case index)
                case = cases[i] # get case nr i. 
                exp_case = self.KB.get(case.explanation) # get Explanations from the cases KB
                exp_test_case = self.KB_test.get(test_case.explanation)
                if(exp_test_case.check_similarity(exp_case)): # if explanation in query at k fit.
                    measurements_dict[n][k] += 1 
            
        measurements_dict_top_k_e = defaultdict(list) # dictionary of lists.
        measurements_dict_top_k_c = defaultdict(list) # dictionary of lists.
        measurements_dict_top_k_n = defaultdict(list) # dictionary of lists.
        measurements_dict_top_k_cp = defaultdict(list) # dictionary of lists.

        # * PERFORM EXPERIMENT
        for n in range(M,N+M,M): # Loop trough the all test_cases.
            # We don't need to add cases to MyCBR in this experiment, but simply only use a section of the CaseBase list at a time.
            cases = init_cases[0:n] # Keep track of the CaseBase

            measurements_dict_top_k_e[n] = [0]*len(cases) # empty list of K elements.
            measurements_dict_top_k_c[n] = [0]*len(cases) # empty list of K elements.
            measurements_dict_top_k_cp[n] = [0]*len(cases) # empty list of K elements.
            measurements_dict_top_k_n[n] = [0]*len(cases) # empty list of K elements.
            
            # We need query every case
            for t_c in test_cases:
                # Query the cases for most similar case.
                query_e = sorted([(t_c.checkEuclidianDistance(c),i) for i,c in enumerate(cases)], key=lambda param: param[0]) # sort by distance
                query_c = sorted([(t_c.checkCosineDistance(c),i) for i,c in enumerate(cases)], key=lambda param: param[0])
                query_cp = sorted([(t_c.checkCosinePrediction(c),i) for i,c in enumerate(cases)], key=lambda param: param[0])
                query_n = [ (0,i) for i in range(0,len(cases)) ] # simply the cases index, unsorted, baseline. 
                # Check if the explanation from query_c top works
                top_case = cases[query_c[0][1]] # get ID of best case
                distance = query_c[0][0]

                # TOP RESULT
                exp_test_case = self.KB_test.get(t_c.explanation)
                exp_val_case = self.KB.get(top_case.explanation)
                if(exp_test_case.check_similarity(exp_val_case)):
                    print("EQUAL:",exp_test_case.exp_map["feature"] ,"==", exp_val_case.exp_map["feature"],"case:",query_c[0][1],"d:", distance)
                
                experiment_4(cases, t_c, query_e, measurements_dict_top_k_e, n)
                experiment_4(cases, t_c, query_c, measurements_dict_top_k_c, n)
                experiment_4(cases, t_c, query_cp, measurements_dict_top_k_cp, n)
                experiment_4(cases, t_c, query_n, measurements_dict_top_k_n, n)

        print("")
        print("Euclidian")
        show_results(measurements_dict_top_k_e, N_T)
        print("Cosine")
        show_results(measurements_dict_top_k_c, N_T)
        print("Cosine Prediction")
        show_results(measurements_dict_top_k_cp, N_T)
        print("None (random)")
        show_results(measurements_dict_top_k_n, N_T)
        

    def run_experiment_5(self, N, N_T, M, unique=True, compress=True):
        """
            ? Test whether we need to present the user with previous cases, aswell as the current explanation.
        """

        np.random.seed(1) # init seed
        #Load the case-base system

        # Init knowledge base for validation data
        self.KB = knowledge_base.KnowledgeBase("exp5")
        self.KB.reset_knowledge() # empty the knowledge-base before we begin.

        self.KB_test = knowledge_base.KnowledgeBase("exp5_test")
        self.KB_test.reset_knowledge() # empty the knowledge-base before we begin.

        # Genererate case objects from these.   
        # N, N_T, unique=False, compress=True
        init_cases, test_cases = self.generate_cases(N, N_T, unique=unique,compress=compress)

        measurements_dict_top_k_e = defaultdict(tuple) # dictionary of lists.
        measurements_dict_top_k_p = defaultdict(int) # dictionary of lists.

        measurements_dict_top_k_c = defaultdict(tuple) # dictionary of lists.
        measurements_dict_top_k_n = defaultdict(tuple) # dictionary of lists.
        measurements_dict_top_k_cp = defaultdict(tuple) # dictionary of lists.
    
        # Randomly check similarity between two cases
        def check_difference(query, test_case, cases, m_dict, m2_dict, n, n_t):
            #k_list = [] # keep track of p,precision scores
            for k, (sim, i) in enumerate(query): # query results, (similarity, case index)
                case = cases[i] # get case nr i. 
                exp_case = self.KB.get(case.explanation) # get Explanations from the cases KB
                exp_test_case = self.KB_test.get(test_case.explanation)

                # Store both values
                p, precision = test_case.checkAnchorFitting(case, preprocess = self.dataman)
                #N_T, n, p,k # Need a matrix per test_query...
                #k_list.append()
                m_dict[n,n_t,k] = (p,precision)
                if(p != 0):
                    m2_dict[n,k] += 1 # how many cases we got a partial match with the case instance

        
        # We want to check similarity between the explanations in the CBR system
        #print(test_cases[1].checkAnchorFitting(init_cases[0], preprocess = self.dataman))

        for n in range(M,N+M,M): # Loop trough the CaseBase batches (different inits).
            cases = init_cases[0:n] # Keep track of the CaseBase

            #measurements_dict_top_k_e[n] = [0]*len(cases) # empty list of K elements.
            #measurements_dict_top_k_c[n] = [0]*len(cases) # empty list of K elements.
            #measurements_dict_top_k_cp[n] = [0]*len(cases) # empty list of K elements.
            #measurements_dict_top_k_n[n] = [0]*len(cases) # empty list of K elements.

            for n_t, t_c in enumerate(test_cases): # Try every test_case against the case_base
                query_e = sorted([(t_c.checkEuclidianDistance(c),i) for i,c in enumerate(cases)], key=lambda param: param[0]) # sort by distance
                query_c = sorted([(t_c.checkCosineDistance(c),i) for i,c in enumerate(cases)], key=lambda param: param[0])
                query_cp = sorted([(t_c.checkCosinePrediction(c),i) for i,c in enumerate(cases)], key=lambda param: param[0])
                query_n = [ (0,i) for i in range(0,len(cases)) ] # simply the cases index, unsorted, baseline. 
                

                check_difference(query_e, t_c, cases, measurements_dict_top_k_e,measurements_dict_top_k_p, n, n_t)

        print(measurements_dict_top_k_e)
        print(measurements_dict_top_k_p)
        print([p for (n,k),p in measurements_dict_top_k_p.items()])

        for n in np.unique([n for n, k in measurements_dict_top_k_p.keys()]): # get unique n keys.
            # TODO: Add zeros to a list and print..
            print([p for (n_i,k),p in measurements_dict_top_k_p.items() if n_i == n])
            
    def run_experiment_6(self, project, jar, storage=False):
        """
            ? Test how the system perform with no initial casebase, and retaining each instance. 
        """

    def run_experiment_x(self):
        """
        
        """
        np.random.seed(1) # init seed
        

def check_contains(element, elements): # CHeck a list of numpy arrays contains a numpy array. 
    for e in elements: # check against everyone in elements.
        if(np.array_equal(e,element)):
            return True
    return False

def check_bool(value):
    if(value == "True"):
        return True
    elif(value == "False"):
        return False
    raise ValueError("Not bool")

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def show_results(dictionary, n):
    for key, item in dictionary.items():
        print("k {} {} {}/{}".format(key, item, sum(item), n))
        # Want to calculate partial sums.
        # 5 added cases at a time.
        partial_sums = []
        for k in range(5,len(item)+5,5):
            partial_sums.append(sum(item[0:k]))
        print(partial_sums)

if __name__ == "__main__":
    ############ Create Argument Parser ############
    parser = argparse.ArgumentParser(description="Experiments controller")

    parser.add_argument("-v","--verbose",default=False, type=bool)

    subparsers = parser.add_subparsers(title="action", dest="experiment", help="experiment to run")

    parser_test = subparsers.add_parser("test")
    parser_rest = subparsers.add_parser("start_server")
    parser_fill = subparsers.add_parser("exp_fill")
    parser_sim = subparsers.add_parser("exp_sim")
    parser_sim_bad = subparsers.add_parser("exp_sim_bad")
    parser_sim_bad2 = subparsers.add_parser("exp_sim_bad2")
    parser_1 = subparsers.add_parser("exp_1")
    #parser_1.add_argument("-N","--num_cases",help="number of cases we initiate with", default=4,
    #                type=check_positive)
    #parser_1.add_argument("-M","--num_retrieval",help="number of queries against the CaseBase (without retain step)", default=4,
    #                type=check_positive)
    parser_2 = subparsers.add_parser("exp_2")
    parser_3 = subparsers.add_parser("exp_3")
    parser_4 = subparsers.add_parser("exp_4")
    parser_5 = subparsers.add_parser("exp_5")


    args = parser.parse_args() # get arguments from command line

    if(args.experiment is None):
        raise ValueError("The experiment arguments required to run, type -h for help")


    experiments = Experiments(verbose=args.verbose)
    # ---------------- Switch between valid experiments for ease of use ------------
    parent = pathlib.Path(__file__).parent # Keep track of folder path of model.
    projects = parent/"CBR"/"projects"
    # Java runnable file of MyCBR REst
    jar = parent/"CBR"/"libs"/"mycbr-rest"/"target"/"mycbr-rest-1.0-SNAPSHOT.jar"
    if(args.experiment == "exp_fill"):
        print("Starting experiment fill with verbose", args.verbose)
        project = projects/"adult_fill"/"adult_fill.prj"
        try:
            experiments.run_experiment_fill(N=50, project=project.absolute(), jar=jar.absolute(), storage=True)
        finally: # Incase the experiment fails for some reason, try to stop the MyCBR rest API server
            experiments.stop_MyCBR()
    elif(args.experiment == "exp_sim"):
        print("Starting experiment sim with verbose", args.verbose)
        project = projects/"adult_sim"/"adult_sim"/"adult_sim.prj"
        try:
            experiments.run_experiment_sim(N_T=2, project=project.absolute(), jar=jar.absolute())
        finally: # Incase the experiment fails for some reason, try to stop the MyCBR rest API server
            experiments.stop_MyCBR()
    elif(args.experiment == "exp_sim_bad"):
        print("Starting experiment sim with verbose", args.verbose)
        project = projects/"adult_sim"/"adult_sim_bad"/"adult_sim_bad.prj"
        try:
            experiments.run_experiment_sim(N_T=2, project=project.absolute(), jar=jar.absolute())
        finally: # Incase the experiment fails for some reason, try to stop the MyCBR rest API server
            experiments.stop_MyCBR()
    elif(args.experiment == "exp_sim_bad2"):
        print("Starting experiment sim with verbose", args.verbose)
        project = projects/"adult_sim"/"adult_sim_bad2"/"adult_sim_bad2.prj"
        try:
            experiments.run_experiment_sim(N_T=11, project=project.absolute(), jar=jar.absolute())
        finally: # Incase the experiment fails for some reason, try to stop the MyCBR rest API server
            experiments.stop_MyCBR()
    elif(args.experiment == "exp_1"): # Test multiple different value combinations.
        N = 40 # number of total cases to test
        N_T = 100 # number of test_cases
        M = 10 # amount of cases we add per test.
        print("Starting experiment 1")
        project = projects/"adult_exp1"/"adult_exp1.prj"
        # For experiment 1, we require a empty case-base, that we fill with cases and explanation.
        try:
            experiments.run_experiment_1(N=N, M=M, N_T=N_T, project=project.absolute(), jar=jar.absolute())
        finally:
            experiments.stop_MyCBR()
    elif(args.experiment == "exp_2"):
        project = projects/"adult_full_s"/"adult_full_s.prj"
        try:
            experiments.run_experiment_2(project=project.absolute(), jar=jar.absolute())
        finally:
            experiments.stop_MyCBR()
    elif(args.experiment == "exp_4"):
        N = 100 # number of total cases to test
        N_T = 100 # number of test_cases
        M = 10 # amount of casebase cases we add per test.
        experiments.run_experiment_4(N=N,N_T=N_T,M=M, unique=True)
    # elif(args.experiment == "full"):
    #     try:
    #         experiments.run_test_full(N=100, project=project.absolute(), jar=jar.absolute())
    #     finally:
    #         experiments.stop_MyCBR()
    elif(args.experiment == "exp_5"):
        N = 10 # number of total cases to test
        N_T = 10 # number of test_cases
        M = 5 # amount of casebase cases we add per test.
        experiments.run_experiment_5(N=N,N_T=N_T,M=M, unique=True)
    elif(args.experiment == "start_server"):
        project = projects/"adult"/"adult.prj"
        # Start Java program in the same terminal, for easy of use
        experiments.start_server(project.absolute(),jar.absolute())
    elif(args.experiment == "test"):
        experiments.run_test()
    
# Eksperiment med å endre på threshold for å se om forklaringen blir bedre med lavere/høyere

# Oppdatere userdefined knowledge i knowledge-basen


# Ulike mål om det passer:
# - Helt identiske
# - 


# ----------------
# Viktigste står først for anchor