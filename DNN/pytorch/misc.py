import numpy as np
import torch
import glob
import os
# Misc functions
def int_to_one_hot_vector(value, size, zero_offset = 0,off_val=0, on_val=1): #
# Size as 
    if int(value-zero_offset) < size and value >= zero_offset:
        v = [off_val] * size
        v[int(value-zero_offset)] = on_val
        return v
    else:
        raise ValueError("Value is greater than size {} < {}".format(value, size))
    # (2,3) -> [0,0,1]    
def int_to_one_hot_vector_rev(value, size, off_value=0, on_val=1):
    v = int_to_one_hot_vector(value,size, off_value, on_val)
    v.reverse()
    return v

def int_to_binary(value,size=2): # int value to convert, size is length of array
    b_array = [int(x) for x in bin(value)[2:]] # Return as smallest binary number
    if(len(b_array) > size): 
        raise ValueError("Impossible to create a binary with size {} for binary({}) {}".format(size,value, b_array))
    if(len(b_array) < size):
        trailing_zeros = size - len(b_array) # check if binary is smaller than we require in digits.
        for zero in range(trailing_zeros):
            b_array.insert(0,0) # Add trailing zeros 01 -> x*0+01
    return b_array   

# Use to represent states and PID, 2 -> [0,1], 1 -> [1,0]
def int_to_binary_rev(value,size=2):
    v = int_to_binary(value, size)
    v.reverse()
    return v

def normalize_array(data, x_min=1):
    """ 
        min_max scaling: x_i = x_i-min(x)/(max(x) - min(x)) 
        z-score : x = x_i - sd/mean(data) 
    """
    #x_max = max(data)    
    #x_min = min(data)
    #data = [(x-x_min)/(x_max-x_min) for x in data]
    mean = sum(data)
    if(mean != 0):
        if(mean != 1.0):
            return [x/mean for x in data]
        return data # already normalized.
    raise ValueError ("Cant normalize an array with no sum")

def min_max_scaling(data):
    x_max = max(data)    
    x_min = min(data)
    return [(x-x_min)/(x_max-x_min) for x in data]

def fix_p(p): # normalize p to sum = 1
        npsum = np.sum(p)
        if npsum != 1.0:
            p = np.multiply(p,1/npsum).tolist()
            #p = p*(1./np.sum(p))
        return p

def all_zeros(data):
    nonzer = np.count_nonzero(data)
    if(nonzer != 0):
        return False
    return True

def int_board_to_network_board(board):
    # * takse an matrix with len = dim*dim
    # Return an tensor with len = dim*dim*2
    board_state = [] # default board.
    for state in board:
        cell_state = int_to_binary_rev(state)
        board_state.extend(cell_state)
    return board_state

def get_legal_states(board): # [1,0,0,1] 1's and 2's
    #return legal moves for every 0's
    legal_moves = []
    for state in board:
        if(state == 0):
            legal_moves.append(1)
        else:
            legal_moves.append(0)
    return np.array(legal_moves)

def find_newest_model(name):
    # Look for network with similar name with different ending. E.g. TESTNET_xxxx
    # Find every file in directory of models with same name
    result = glob.glob("models/"+name+"/"+name+"*")
    if(len(result) == 0):
        return None
    result_2 =[x.split("_") for x in result]
    # We need to use the second index to rank results.
    result_3 = max(result_2,key=lambda x:int(x[1]))
    path = "_".join(result_3)
    if(os.path.isfile("_".join(result_3))):
        return path
    else:
        print("{} is not a regular file".format(path))

def find_models(name):
    # Find every file in directory of models with same name
    result = glob.glob("models/"+name+"/"+name+"*")
    if(len(result) == 0):
        return None
    return result

def get_player_states(board,dim,ravel=True): # Takes in an 5x5 array and outputs two 5x5 arrays with 1 where player played.
    if( type(board) == list): # Need to translate to np.array
        board = np.array(board) # Hopefulle, dimentions are the same.
    board = np.reshape(board,(dim,dim))
    
    player1_board = np.in1d(board,1).reshape(board.shape).astype(int)
    #print(player1_board)
    player2_board = np.in1d(board,2).reshape(board.shape).astype(int)
    if(ravel):
        return player1_board.ravel(), player2_board.ravel()
    return player1_board, player2_board

#   print(get_player_states([0,1,1,2,2,0,1,1,2,2,0,1,1,2,2,0,1,1,2,2,0,1,1,2,2], dim=5,ravel=False))

def one_hot_array(board): # turn [1,2,1] -> [1,0,0,1,1,0]
    board_array = []
    for cell in board:
        one_hot = int_to_binary_rev(cell,2)
        board_array.extend(one_hot)
    return board_array # return as regular list.

# Data translations, to handle different network inputs.
def get_cnn_input(data_pid, data_inputs, dim): # This can be a list of lists.
    PID = np.array([np.full((dim,dim),(2-pid)) for pid in data_pid]) # Player 2 = 0, Player 1 = 1
    inputs = np.array([get_player_states(board,dim,ravel=False) for board in data_inputs])
    PID = np.reshape(PID,(PID.shape[0],1,PID.shape[1],PID.shape[2]))
    inputs = np.concatenate((inputs,PID),axis=1)
    return torch.from_numpy(inputs).float() # * (B x 3 x 5 x 5) for CNN2d

def get_normal_input(data_pid,data_inputs): # Don't need to know board dimentions.
    # inputs:  PID + board_state
    PID =  np.array([int_to_binary_rev(pid) for pid in data_pid])
    inputs = np.array([one_hot_array(board) for board in data_inputs])
    inputs = np.append(PID, inputs, axis=1)
    # * PID + board_state as one hot vectors per cell.
    return torch.from_numpy(inputs).float() # * (Bx52) 

def get_normal_2(data_pid, data_inputs): #Return as 26 x 25.
    PID = np.array([reverse_2([pid]) for pid in data_pid])
    inputs = np.array([reverse_2(board) for board in data_inputs])
    inputs = np.append(PID, inputs, axis=1)
    return torch.from_numpy(inputs).float() # * (Bx26)

def reverse_2(data):#max(result_2,key=lambda x:int(x[1]))
    for i,n in enumerate(data):
        if( n == 2):
            data[i] = -1 # Rest should be the same
    return data