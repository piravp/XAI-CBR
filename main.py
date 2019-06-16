""" 
    This is the main file used to reproduce results presented in the thesis.
    By seeding the randomization at each step the results are guaranteed to be reproduceable.

    Only tested on Windows OS, anything else could fail at the MyCBR shutdown step.

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
from DNN import explanation_base, explanation
from DNN.explanation import Explanation
from CBR.src import CBRInterface                            # Interface against myCBR's REST API
from CBR.src.case import Case
# Integrated gradients
from deepexplain.tensorflow import DeepExplain
from keras import backend as K
from keras.models import Model

from collections import defaultdict
#import operator
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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

        #* Init integrated gradients variable
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
            time.sleep(5) # 5 seconds
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
        """
            Get attribution from one instances using Integraded Gradients. 
        """
        with DeepExplain(session=K.get_session()) as de: # Init session, to keep gradients.
            input_tensors = self.bb.model.inputs
            output_layer = self.bb.model.outputs
            fModel = Model(inputs=input_tensors, outputs=output_layer)
            target_tensor = fModel(input_tensors)

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
        """
            Get attribution from multiple instances using Integraded Gradients. 
        """
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

    def get_cases(self, instances, encoding, EB, compress): 
        attributions = self.get_attribution_multiple(encoding, compress)
        explanations, predictions = self.get_explanation_prediction(encoding)

        case_objects = [] # list of cases
        for i, inc in enumerate(instances):  
            exp_id = EB.add_knowledge(explanations[i])
            #TODO: clean up the case_generation (not a very good approach ..)
            case_objects.append(Case(age=inc[0], workclass=inc[1], education=inc[2], martial_status=inc[3], occupation=inc[4],
                relationship=inc[5], race=inc[6], sex=inc[7], capital_gain=inc[8], capital_loss=inc[9],
                hours_per_week=inc[10],country=inc[11],
                weight=attributions[i], prediction = predictions[i], explanation = exp_id, EB=EB))
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

        cases = self.get_cases(instances = init_cases, encoding=init_cases_enc, EB = self.EB, compress=compress)
        cases_test = self.get_cases(instances = test_cases[:N_T], encoding=test_cases_enc[:N_T], EB = self.EB_test, compress=compress)

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
        np.random.seed(99) 

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

        # Create explanation-base
        print('Checking explanation-base...')
        self.EB = explanation_base.ExplanationBase("final")   # Init knowledge base for validation data
        self.EB.reset_knowledge() # empty the explanation-base before we begin.
        self.EB_test = explanation_base.ExplanationBase("final_test")   # Init knowledge base for test data (not used but need to be init)
        self.EB.reset_knowledge() # empty the explanation-base before we begin.


        # Genererate case objects from these.   
        cases, _ = self.generate_cases(N=N, N_T=1)
        self.EB_test.reset_knowledge() # empty the explanation-base for the test (required to have N_T > 0)

        for c in cases:
            print(c, '\n')

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
            # 1. Retreve cases from CB
            # 2. Perform k most similar and find most similar cases
            # 3. Retrieve explanation for most similar case
            # 4. Present explanation to user
        """

        # Init same seed every time for consistency
        np.random.seed(99) 

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
        print('Checking explanation-base...')
        self.EB = explanation_base.ExplanationBase("final")
        # self.EB.reset_knowledge() # empty the explanation-base before we begin.

        self.EB_test = explanation_base.ExplanationBase("final_test")
        self.EB_test.reset_knowledge() # empty the explanation-base before we begin.


        _, test_cases = self.generate_cases(N=1, N_T=N_T)
        # test_cases = self.generate_test_cases(N_T) 
        print('# of cases from test set:', len(test_cases))
    

        # single_test_case = test_cases[0]

        for i in range(0, N_T):
            print(">>>>>>>>> Adding test case {} <<<<<<<<<<<<<".format(i))
            t_case = test_cases[i]
            # Show query case
            # print(t_case)
            self.retrieve(testCase=t_case, topK=k)
            print('\n\n')

    
    # Helper used in run_retrieve
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
        print(df)
        # Print explanation for test case
        print('----EXPLANATION FOR TEST CASE:----')
        print('caseID', caseID)
        testExpId = testCase.explanation 
        exp = self.EB_test.get(testExpId)
        print(exp.get_explanation(self.dataset.feature_names,self.dataset.categorical_names), '\n')

        # Explanation for cases in cb
        print('----MOST SIMILAR CASE:----')  
        for casename, row in df.iterrows():
            # Explanation for val case (in case-base)
            print('*', casename, row.to_string())
            res = self.CBR.getSingleInstance(conceptID='Person', casebaseID='cb0', instanceID=casename)
            res = res["case"]
            # The case itself
            # print(res)
            c = Case( res['Age'], res['CapitalGain'], res['CapitalLoss'], res['Country'], res['Education'], 
                       res['Explanation'], res['HoursPerWeek'], res['MaritalStatus'], res['Occupation'],
                        res['Prediction'], res['Race'], res['Relationship'], res['Sex'], res['Weight'], 
                         res['Workclass'], self.EB)
            
            valExpId = c.explanation #int(res["case"]["Explanation"])
            exp_val = self.EB.get(valExpId)
            # Precision, coverage, etc
            print(explanation.Explanation(**exp_val.exp_map))
            # Human readable rules
            print("\nEXPLANATION:")
            print(exp_val.get_explanation(self.dataset.feature_names,self.dataset.categorical_names))
            # Delete case from case-base after getting explanation, so next test-case is not affected
            # (Note that even though the addition of test-cases are not persistent, the case is going to 
            # persist for as long as the sessions lifetime lasts. Hence, this command is needed.)
            self.CBR.deleteInstance(casebaseID='cb0', conceptID='Person', instanceID=casename)

            # partial = testCase.checkSimilarityPartialExplanation(c)
            # print('Partial', partial)

    def run_weight_test(self, N, N_T, M, unique=True, compress=True):
        """
            ? Test whether the attribution score can be used for retrieval alone on the CaseBase

        """
        ############# initializing #############
        np.random.seed(1) # init seed

        # Init explanation base for validation data
        self.EB = explanation_base.ExplanationBase("epx_weight")
        self.EB.reset_knowledge() # empty the explanation-base before we begin.

        self.EB_test = explanation_base.ExplanationBase("epx_weight_test")
        self.EB_test.reset_knowledge() # empty the explanation-base before we begin.

        # Genererate case objects from these.   
        # N, N_T, unique=False, compress=True
        print("Init generating cases, this should take approximatly", N+N_T,"seconds")
        t = time.time()
        init_cases, test_cases = self.generate_cases(N, N_T, unique=unique,compress=compress)
        ################### STARTING EXPERIMENT ###################
        print("Generating", N, "cases and", N_T, "test cases completed in", time.time()-t,"seconds")
        #print("Cosine Similarity, self:",test_cases[0].checkCosineDistance(test_cases[0]))
        #print("Eucluidian similarity, self:", test_cases[0].checkEuclidianDistance(test_cases[0]))

        # Want to generate a plot per case_base fill instance. One with 5,10,15,20,25,30 etc.

        def experiment_4(cases, test_case, query, measurements_dict, n): # n is number of cases in casebase
            for k, (sim, i) in enumerate(query): # query index, (similarity, case index)
                case = cases[i] # get case nr i from CaseBase
                exp_case = self.EB.get(case.explanation) # get Explanations from the cases EB
                exp_test_case = self.EB_test.get(test_case.explanation)
                
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
                # Query the cases for most similar case ranking and sort distances
                query_e = sorted([(t_c.checkEuclidianDistance(c),i) for i,c in enumerate(cases)], key=lambda param: param[0])
                query_c = sorted([(t_c.checkCosineDistance(c),i) for i,c in enumerate(cases)], key=lambda param: param[0])
                query_cp = sorted([(t_c.checkCosinePrediction(c),i) for i,c in enumerate(cases)], key=lambda param: param[0])
                query_n = [ (0,i) for i in range(0,len(cases)) ] # simply the cases index, unsorted, baseline. 

                # Run test on each measurment dictionary to get results.
                experiment_4(cases, t_c, query_e, measurements_dict_top_k_e, n)
                experiment_4(cases, t_c, query_c, measurements_dict_top_k_c, n)
                experiment_4(cases, t_c, query_cp, measurements_dict_top_k_cp, n)
                experiment_4(cases, t_c, query_n, measurements_dict_top_k_n, n)
        
        # what should be drawn and properties regarding each plot
        measurements = [measurements_dict_top_k_e, measurements_dict_top_k_c, measurements_dict_top_k_cp, measurements_dict_top_k_n]
        lines = ['mo','cD','g^','r*']
        size = [10,10,10,10]
        fig, subs = plt.subplots(len(measurements_dict_top_k_e.keys()), 1) # create one subplot for each case-base test.
        

        fig.suptitle('Test against 50 test cases, +10 in Case-Base per graph', fontsize=13)
        for i,sub in enumerate(subs): # itterate over the subplots
            for c, measurement in enumerate(measurements): # loop though measurment data and calculate top k slices.
                partial_sums = [] # top k cases
                for k in range(M,len(measurement[(i+1)*M])+M,M): # calculate slices
                    partial_sums.append(sum(measurement[(i+1)*M][0:k])) # get slices from measurements.
                sub.plot(range(M,len(partial_sums)*M+1,M), partial_sums, lines[c],markersize=size[c]) # center line

            sub.set_xticks(range(M,len(partial_sums)*M+1,M)) # set ticks to number of cases in each test.
            sub.set_ylabel("hits") 
            sub.set_ylim(bottom=0) 
            sub.yaxis.set_major_locator(MaxNLocator(integer=True)) # create good y-axis labels.
        fig.legend(('Euclidian', 'Cosine','CosinePred','None(random)'), loc='upper right') # mark each plot
        plt.xlabel("hits in top k results from query") 
        plt.show()

def check_contains(element, elements): # CHeck a list of numpy arrays contains a numpy array. 
    for e in elements: # check against everyone in elements.
        if(np.array_equal(e,element)):
            return True
    return False

if __name__ == "__main__":
    ############ Create Argument Parser ############
    parser = argparse.ArgumentParser(description="Main controller")
    parser.add_argument("-v","--verbose",default=False, type=bool)
    subparsers = parser.add_subparsers(title="action", dest="main", help="main to run")

    fill_final = subparsers.add_parser("fill_final")
    retrieve = subparsers.add_parser("retrieve")
    test_weight = subparsers.add_parser("test_weight")

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
    elif(args.main == "test_weight"):
        N = 50 # number of total cases to test against
        N_T = 50 # number of test_cases
        M = 10 # amount of casebase cases we add per test.
        main.run_weight_test(N=N,N_T=N_T,M=M, unique=True)