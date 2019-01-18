import network
import Datamanager
import pandas as pd

def read_data_pd(name,columns,encoding="latin-1"):
    data = pd.read_csv(name,delimiter=",",encoding=encoding) # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe5 in position 38: invalid continuation byte
    df = pd.DataFrame(data=data) # collect panda dataframes
    return df

def test_network():
    wine_attributes = [""]
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
    df = read_data_pd("../Data/wine.csv",columns = columns)

    print(df.head())
    df.columns = columns
    #Cov.columns = ["Sequence", "Start", "End", "Coverage"]
    print(df.head())
    dataman = Datamanager.Datamanager(data=df)   

    #model = network.NN_25("whine",dim=10)  

test_network()