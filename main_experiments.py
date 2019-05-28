""" 
    This is the main file used for the experimentation.
    By seperating the experiments and seeding the randomization at each step, it will be reproduceable.

    Its important that the myCBR rest projet is running.
    # COMMAND TO START PROGRAM (in CBR/libs/mycbr-rest)
    > java -DMYCBR.PROJECT.FILE=<absolute_path>\XAI-CBR\CBR\projects\adult\adult.prj -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar

    For more Info see README file.
    
    # Better Comments extension (VS code) used for easier reading of different comments.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

# set seed from numpy to 1.
from DNN.kera import network # Import black-box (ANN)
from DNN.kera import pre_processing # Import dataset preprocessing
from DNN.Induction.Anchor import anchor_tabular, utils # explanatory framework
from DNN import knowledge_base,explanation
from CBR.src import CBRInterface

class experiments():
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

        #* Init BlackBox
        self.bb = network.BlackBox(name="NN-adult-5",c_path="NN-Adult-5/NN-Adult-5-8531.hdf5")
        if(verbose):
            self.bb.evaluate(data_train=self.anchors_explainer.encoder.transform(self.dataset.data_train).toarray(),train_labels=self.dataset.train_labels,
                    data_test=self.anchors_explainer.encoder.transform(self.dataset.data_test).toarray(),test_labels=self.dataset.test_labels)
        
        # Init IntGrad

        # Init KnowledgeBase

        # Init CaseBase
        self.CBR = CBRInterface.RESTApi() # load CBR restAPI class, for easy access.
        pass
    
    def experiment_1(self,N,M,project): # N is number of cases in casebase, M is number of retrievals
        """ 
            ? Test whether or not we are able to use previous explanations in the CBR system 
            
            Pre initiate the case-base with N randomly selected cases from the validation set
            Preform M random retrievals and check if an explanation can be found.

            Count how many explanations are correct, with respect to the explanation generated for the specific instance.
        """
        np.random.seed(1) # init seed
        #Load the case-base system
        # java -DMYCBR.PROJECT.FILE=/path/to/project.prj -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar
        os.system("java -DMYCBR.PROJECT.FILE={} -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar".format(project))
        #Initiate cases into the project
        #exp = experiments(verbose=True)

        #os.system("java -{}".format("help"))
        p = Popen(["java","-help"])
        #p.terminate()
        print("test")

        # java -DMYCBR.PROJECT.FILE=/path/to/project.prj -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar
        os.system("java -DMYCBR.PROJECT.FILE={} -jar ./target/mycbr-rest-1.0-SNAPSHOT.jar".format(project))
        #Initiate cases into the project
        pass

    def experiment_1_sim(self):
        """
            ? Test different similarity measures against the CaseBase


        """
        

    def experiment_2(self):
        """  
            ? Test wheter or not we are able to use previous explanations in tandom with custom explanations given by a domain expert.

            Pre Initiate the knowledge-base with custom explanation anchors, to explain a given case-instance. 
            If the expert knowledge fit a new problem, then it is used instead of from the case-base. 

        """
        np.random.seed(1) # init seed
        pass

    def experiment_3(self):
        """
            ? Test whether the attribution score from integradet gradients can be used to help with the retrieval of relevant cases 

        """
        np.random.seed(1) # init seed

    def experiment_4(self):
        """
            ? Test whether the attribution score can be used for retrieval alone on the case-base

        """
        np.random.seed(1) # init seed
        # Init the case-base

        # pre initiate 

    def experiment_5(self):
        """
            ? Test whether we need to present the user with previous cases, aswell as the current explanation.
            
        """
        np.random.seed(1) # init seed
    

    def experiment_x(self):
        """
        
        """
        np.random.seed(1) # init seed
        pass

def exp_case_base_size():
    """ Test initializing casebase of different sizes and querying new problems. """
    np.random.seed(1) # init seed

    dataman = pre_processing.Datamanager(dataset="adults",in_mod="normal",out_mod="normal")
    dataset = dataman.ret

    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data_train, dataset.categorical_names)
        
    # ! Explainer.encoder.transform return sparse matrix, instead of dense np.array
    explainer.fit(dataset.data_train, dataset.train_labels, 
                dataset.data_validation, dataset.validation_labels) # Fit the labels to the explainer.
    print(network)

    bb = network.BlackBox(name="NN-adult-5",c_path="NN-Adult-5/NN-Adult-5-8531.hdf5")

    bb.evaluate(data_train=explainer.encoder.transform(dataset.data_train).toarray(),train_labels=dataset.train_labels,
                    data_test=explainer.encoder.transform(dataset.data_test).toarray(),test_labels=dataset.test_labels)

    exit()
    # Check if REST api is running with project.
    CBR = CBRInterface.RESTApi() # load CBR restAPI class, for easy access.


exp = experiments(verbose=True)