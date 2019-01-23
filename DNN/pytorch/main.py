import network
import Datamanager
import pandas as pd

import torch.optim as optim
import torch.nn.modules.loss as t_loss

import os,sys,inspect
# add parent folder to path ( DNN )
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

def read_data_pd(name,columns,encoding="latin-1"):
    data = pd.read_csv(name,delimiter=",",encoding=encoding) # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe5 in position 38: invalid continuation byte
    df = pd.DataFrame(data=data) # collect panda dataframes
    return df

def test_network():
    wine_attributes = ["alch","malic","ash","alcash","mag","phen","flav","nfphens","proant","color","hue","dil","prol"]
    columns = ["class","alch","malic","ash","alcash","mag","phen","flav","nfphens","proant","color","hue","dil","prol"]
    """
        0) class
    	1) Alcohol
        2) Malic acid
        3) Ash
        4) Alcalinity of ash  
        5) Magnesium
        6) Total phenols
        7) Flavanoids
        8) Nonflavanoid phenols
        9) Proanthocyanins
        10)Color intensity
        11)Hue
        12)OD280/OD315 of diluted wines
        13)Proline    
    """

    from Induction.IntGrad.integratedGradients import random_baseline_integrated_gradients, integrated_gradients
    exit()

    df = read_data_pd("../../Data/wine.csv",columns = columns)

    df.columns = columns # Add columns to dataframe.
    #Cov.columns = ["Sequence", "Start", "End", "Coverage"]
    dataman = Datamanager.Datamanager(dataframe_train=df,classes=3,dataset="wine")   

    model = network.NN_3_25("wine",in_dim=13,out_dim=3)
    print(model.input_type)
    optimizer = optim.Adam(model.parameters(), lr=0.01,betas=(0.9,0.999),eps=1e-6)
    #loss = network.RootMeanSquareLoss()
    #loss = t_loss.L1Loss()
    loss = t_loss.MSELoss()
    network.train(model, dataman,validation=True, optimizer = optimizer,loss_function = loss, batch=20, iterations=50)

    # model is trained, we want to figure out the attribute distributions.

    

def test_accuracy():
    import numpy as np
    import torch
    import torch.nn.functional as F
    #nn.Softmax(x)
    #x = torch.rand(4,5)
    x = torch.tensor(np.array([[0.5, 0.4, 0.1, 0.2], 
                                [0.0, 0.7, 0.6, 0.8],
                                [0.1, 0.2, 0.3, 0.5], 
                                [0.1, -0.2, 1.20, 0.05]])).float()
    y = torch.from_numpy(np.array([[0],[3],[3],[1]])).long() # shape = [4,1] 
    print(x)
    print(y)
    print(F.softmax(x,dim=-1))
    result = torch.topk(x,1)[1] # only get best
    #print(x.scatter(1,indices,topk))
    print(y,"\n",result)
    correct = result.eq(y) # 1 where equal elements.
    print(correct)
    correct = int(torch.sum(correct))

    print(result.shape[0])

    total_num = result.shape[0]

    print((100*correct/total_num))