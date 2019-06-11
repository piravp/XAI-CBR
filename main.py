""" 
    This is the main file used to reproduce results presented in the thesis.
    By seeding the randomization at each step the results are guaranteed to be reproduceable.

    ONLY TESTED IN WINDOWS

    For more Info see README file.
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
from DNN.kera import network                                # Import black-box (ANN)
from DNN.kera import pre_processing                         # Import dataset preprocessing
from DNN.Induction.Anchor import anchor_tabular, utils      # Explanatory framework (anchor)
from DNN import knowledge_base, explanation
from CBR.src import CBRInterface                            # Interface against myCBR's REST API
from CBR.src.case import Case
# Integrated gradients
from deepexplain.tensorflow import DeepExplain
from keras import backend as K
from keras.models import Model

from collections import defaultdict

class Main():
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
        print("Starting myCBR REST server") # ,stdout=PIPE,stderr=PIPE
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
        print("------------------- MyCBR is ready -------------------")

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


    # ----------------------------------------------------------------------------------------------------- #
    #                               F I N A L   I N   R E P O R T                                           #
    #          ************************************************************************                     #
    #            These are the commands that can be run to reproduce results in report                      #
    # ----------------------------------------------------------------------------------------------------- #
    def run_fill_final(self, N, project, jar, storage):
        """
            Populate case-base with N cases (with cases from validation set).
        """
        # Init random seed for consistency
        np.random.seed(2) 

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
        self.KB = knowledge_base.KnowledgeBase("final")   # Init knowledge base for validation data
        self.KB.reset_knowledge() # empty the knowledge-base before we begin.
        self.KB_test = knowledge_base.KnowledgeBase("final_test")   # Init knowledge base for test data (not used but need to be init)
        self.KB.reset_knowledge() # empty the knowledge-base before we begin.


        # Genererate case objects from these.   
        cases, _ = self.generate_cases(N=N, N_T=1)
        self.KB_test.reset_knowledge() # empty the knowledge-base for the test (required to have N_T > 0)

        
        # Add all of the cases (from validation set) to the case-base
        # NOTE! HTTP header is limited to about 10 cases, so adding cases need to be done in batches of max 10.
        batch_size = 10
        cases_batch = self.divide_batches(l=cases, n=batch_size)
        for i, batch in enumerate(cases_batch):
            print('Adding batch {} with batch size {}...'.format(i, len(batch)))
            print(self.CBR.addInstancesCases(casebaseID='cb0', conceptID='Person', cases=batch))
            print()

   

    def run_retrieve(self, N_T, k, project, jar, storage):
        """
            Retrieve k most similar cases
        """

        # Init same seed every time for consistency
        np.random.seed(2) 

        # Start CBR project
        self.start_MyCBR(project, jar, storage) 
        self.myCBR_running() # Continue running.

        # Get the concept ID, and CaseBase ID
        conceptID = self.CBR.getConceptID()
        casebaseID = self.CBR.getCaseBaseID()

        # Check whether or not the casebase is filled with cases
        size = self.CBR.getCaseBaseSize(conceptID = conceptID, casebaseID = casebaseID)
        if(size == 0):
            raise ValueError("Case-base is empty")


        # Init knowledge base for validation data
        print('Checking knowledge-base...')
        self.KB = knowledge_base.KnowledgeBase("final")
        # self.KB.reset_knowledge() # empty the knowledge-base before we begin.

        self.KB_test = knowledge_base.KnowledgeBase("final_test")
        self.KB_test.reset_knowledge() # empty the knowledge-base before we begin.


        _, test_cases = self.generate_cases(N=1, N_T=N_T)
        # test_cases = self.generate_test_cases(N_T) 
        print('# of cases from test set:', len(test_cases))
    

        # single_test_case = test_cases[0]

        for i in range(0, N_T):
            print(">>>>>>>>> Adding test case {} <<<<<<<<<<<<<".format(i))
            t_case = test_cases[i]
            self.retrieve(testCase=t_case, topK=k)
            print('\n\n')

        # self.CBR.retrieve_k_sim_byID(conceptID=conceptID, casebaseID=casebaseID, queryID='', k=5)
        # 1. Retreve cases from CB
        # 2. Perform k most similar and find most similar cases
        # 3. Retrieve explanation for most similar case
        # 4. Present explanation to user

    
    def retrieve(self, testCase, topK):
        """
            As the retrieve function implemented in myCBR's REST API does not support sending a query, 
            we need to add the test case to the case-base temporarily in order to find the most similar case.
            Note that there are no persistent effects as myCBR is run with save(storage) flag set to false
        """
        # Add single test case
        caseID = self.CBR.addInstancesCases(casebaseID='cb0', conceptID='Person', cases=[testCase])
        caseID = eval(caseID)[0]    # Convert from string to list and get only item


        df = self.CBR.retrieve_k_sim_byID(conceptID='Person', casebaseID='cb0', queryID=caseID, k=topK)
        df = df.iloc[1:]            # Exclude the test-case itself which is also returned as it is now part of the cb

        # Print explanation for test case
        print('----EXPLANATION FOR TEST CASE:----')
        print('caseID', caseID)
        testExpId = testCase.explanation 
        exp = self.KB_test.get(testExpId)
        print(exp.get_explanation(self.dataset.feature_names,self.dataset.categorical_names), '\n')

        # Explanation for cases in cb
        print('----TOP K MOST SIMILAR CASES:----')  
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
            # Delete case from case-base after getting explanation, so next test-case is not affected
            # (Note that even though the addition of test-cases are not persistent, the case is going to 
            # persist for as long as the sessions lifetime lasts. Hence, this command is needed.)
            self.CBR.deleteInstance(casebaseID='cb0', conceptID='Person', instanceID=casename)

            # partial = testCase.checkSimilarityPartialExplanation(c)
            # print('Partial', partial)


if __name__ == "__main__":
    ############ Create Argument Parser ############
    parser = argparse.ArgumentParser(description="Main controller")
    parser.add_argument("-v","--verbose",default=False, type=bool)
    subparsers = parser.add_subparsers(title="action", dest="main", help="main to run")



    fill_final = subparsers.add_parser("fill_final")
    retrieve = subparsers.add_parser("retrieve")

    args = parser.parse_args() # Get arguments from command line


    if(args.main is None):
        raise ValueError("The main arguments are required to run. Type -h for help")

    main = Main(verbose=args.verbose)
    parent = pathlib.Path(__file__).parent # Keep track of folder path of model.
    projects = parent/"CBR"/"projects"
    # Java runnable file of myCBR REST
    jar = parent/"CBR"/"libs"/"mycbr-rest"/"target"/"mycbr-rest-1.0-SNAPSHOT.jar"
    if(args.main == "fill_final"):
        project = projects/"adult_final"/"adult_final.prj"
        try:
            main.run_fill_final(N=5, project=project.absolute(), jar=jar.absolute(), storage=True)
        finally: # In case the command fails for some reason, try to stop the MyCBR rest API server
            main.stop_MyCBR()
    elif(args.main == "retrieve"):
        project = projects/"adult_final"/"adult_final.prj"
        try:
            main.run_retrieve(N_T=1, k=1, project=project.absolute(), jar=jar.absolute(), storage=False)
        finally: # In case the command fails for some reason, try to stop the MyCBR rest API server
            main.stop_MyCBR()