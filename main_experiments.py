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
import os
import signal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import numpy as np
import subprocess
from subprocess import Popen,CREATE_NEW_CONSOLE,PIPE
import time

import argparse

# set seed from numpy to 1.
from DNN.kera import network # Import black-box (ANN)
from DNN.kera import pre_processing # Import dataset preprocessing
from DNN.Induction.Anchor import anchor_tabular, utils # explanatory framework
from DNN import knowledge_base,explanation
from CBR.src import CBRInterface,case
# Integraded gradients
from deepexplain.tensorflow import DeepExplain
from keras import backend as K
from keras.models import Model

class Experiments():
    def __init__(self, verbose=False):
        #* Init Dataset
        dataman = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
        self.dataset = dataman.ret
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
        
        # Init IntGrad

        # Init KnowledgeBase

        # Init CaseBase
        self.CBR = CBRInterface.RESTApi() # load CBR restAPI class, for easy access.

    def get_attribution(self,instance):
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
            attribution = de.explain('intgrad',target_tensor, input_tensors, explainer.encoder.transform(instance).toarray())
            one_hot_vector = attribution[0][0]

            # Compress one_hot_encoded input vector attribution into one per original attribute. (71 -> 12) 
            start = 0
            for n in self.n_values: # itterate over each categorie
                compressed = sum(one_hot_vector[start:start+n])
                attribution_weights_instance.append(compressed)
                start += n # increase start slice

        return attribution_weights_instance # return list of attributions.


    def start_MyCBR(self, project, jar, storage=False): # Start myCBR project file # TO put everything into the same console, remove flag.
        print("Starting myCBR Rest server") # ,stdout=PIPE,stderr=PIPE
        self.process = Popen(["java","-DMYCBR.PROJECT.FILE={}".format(project),
                            "-Dsave={}".format(storage),"-jar",str(jar)],shell=True)#, creationflags=CREATE_NEW_CONSOLE)
        # Return once it is up and running

    def myCBR_running(self):
        status = self.CBR.checkStatus()
        while(status == 404 or status == 500): # if not sucessfull, keep sleeping
            time.sleep(5)
            status = self.CBR.checkStatus()
            print("Current status code myCBR:",status)
        print("------------MyCBR ready------------")

    def stop_MyCBR(self):
        if(self.process.poll() is None): # check if process is still running.
            print("KILLING RUNNING PROCESSES BEFORE EXITING")
            if os.name == 'nt': # teriminate doest work on windows.
                subprocess.call(['taskkill','/F','/T','/PID',str(self.process.pid)])
            self.process.terminate() # terminalte process
            


    ########################################################################
    # **************************** EXPERIMENTS *****************************
    ########################################################################

    def run_experiment_sim(self, project, jar, storage=False):
        """
            ? Test different similarity measures against the CaseBase
            
            Fill the case-base with cases, and corresponding knowledge in the knowledge-base

            Query the CaseBase with the different similarity measurements made.

            Test firstly using only the similarty measure itself ( returns a float for each case)

            Finally test wether or not the attribution could improve uppon the similarity measurement.

        """

        # Start the caseBase 
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
        print("size", size)
        if(size == 0):  # We need to fill the CaseBase with cases
            # Lets fill it with 100 cases
            # And get the explanation from each.



            pass

        # Perform different similarty measurments.



    def run_experiment_1(self, N, M, project, jar, storage=False): # N is number of cases in casebase, M is number of retrievals
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

        # INIT EXPERIMENT:
        
        # INIT Knowledge base
        #
        self.KB = knowledge_base.KnowledgeBase("exp1")
        self.KB.reset_knowledge() # empty the knowledge-base before we begin.

        # Randomly select N from validation set.
        
        # Randomly select M from test set to check against.

        # Fill the case-base with cases.

        

    def run_experiment_2(self,project, jar, storage=False):
        """  
            ? Test wheter or not we are able to use previous explanations in tandom with custom explanations given by a domain expert.

            Pre Initiate the knowledge-base with custom explanation anchors, to explain a given case-instance. 
            If the expert knowledge fit a new problem, then it is used instead of from the case-base. 

        """
        np.random.seed(1) # init seed

    def run_experiment_3(self,project, jar, storage=False):
        """
            ? Test whether the attribution score from integradet gradients can be used to help with the retrieval of relevant cases 

        """
        np.random.seed(1) # init seed

    def run_experiment_4(self,project, jar, storage=False):
        """
            ? Test whether the attribution score can be used for retrieval alone on the case-base

        """
        np.random.seed(1) # init seed
        # Init the case-base

        # pre initiate 

    def run_experiment_5(self,project, jar, storage=False):
        """
            ? Test whether we need to present the user with previous cases, aswell as the current explanation.
            
        """
        np.random.seed(1) # init seed
    

    def run_experiment_x(self):
        """
        
        """
        np.random.seed(1) # init seed
        
    def start_server(self, project, jar, storage=False):
        # Simply start the Server, and dont stop running
        self.start_MyCBR(project=project,jar=jar)
        self.myCBR_running() # Continue running.

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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Experiments controller")

    parser.add_argument("-v","--verbose",default=False, type=bool)
    #parser.add_argument("-s","--storage",default=False, type=bool)

    subparsers = parser.add_subparsers(title="action", dest="experiment", help="experiment to run")

    parser_rest = subparsers.add_parser("start_server")

    parser_sim = subparsers.add_parser("exp_sim")

    parser_1 = subparsers.add_parser("exp_1")
    parser_1.add_argument("-N","--num_cases",help="number of cases we initiate with", default=4,
                    type=check_positive)
    parser_1.add_argument("-M","--num_retrieval",help="number of queries against the CaseBase (without retain step)", default=4,
                    type=check_positive)

    parser_2 = subparsers.add_parser("exp_2")

    parser_3 = subparsers.add_parser("exp_3")

    parser_4 = subparsers.add_parser("exp_4")

    parser_5 = subparsers.add_parser("exp_5")


    args = parser.parse_args() # get arguments from command line

    if(args.experiment is None):
        raise ValueError("The arguments required to run, type -h for help")


    experiments = Experiments(verbose=args.verbose)

    # Switch between the valid experiments

    parent = pathlib.Path(__file__).parent # Keep track of folder path of model.
    projects = parent/"CBR"/"projects"
    # Java runnable file of MyCBR REst
    jar = parent/"CBR"/"libs"/"mycbr-rest"/"target"/"mycbr-rest-1.0-SNAPSHOT.jar"
    if(args.experiment == "exp_sim"):
        print("Starting Experiment sim with verbose", args.verbose)
        project = projects/"adult_sim"/"adult_sim.prj"
        try:
            experiments.run_experiment_sim(project=project.absolute(), jar=jar.absolute())
        finally: # Incase the experiment fails for some reason, try to stop the MyCBR rest API server
            experiments.stop_MyCBR()
    elif(args.experiment == "exp_1"): # Test multiple different value combinations.
        N = [2,4,6,8,16,32,64,128,256]
        M = [2,3,6,8,16,32,64,128,2560]
        print("Starging Experiment 1 with num_cases = , num_retrievals = ".format(args.num_cases, args.num_retrieval))
        project = projects/"adult2-test"/"adult2.prj"
        # For experiment 1, we require a empty case-base, that we fill with cases and explanation.
        try:
            experiments.run_experiment_1(N=args.num_cases, M=args.num_retrieval, project=project.absolute(), jar=jar.absolute())
        finally:
            experiments.stop_MyCBR()
    elif(args.experiment == "exp_2"):
        try:
            experiments.run_experiment_2(N=args.num_cases, M=args.num_retrieval, project=project.absolute(), jar=jar.absolute())
        finally:
            experiments.stop_MyCBR()
    elif(args.experiment == "start_server"):
        project = projects/"adult"/"adult.prj"
        experiments.start_server(project.absolute(),jar.absolute())
    
