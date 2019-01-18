import torch
import torch.nn as nn
import torch.nn.init as init
import Datamanager
import os

class Model(nn.Sequential):
    def __init__(self, *args,name="Network", filepath=None, input_type=1):
        #print("args",args)
        super().__init__(*args) # Pass rest of network to parent, to create the network with Sequential.
        self.name = name
        self.input_type = input_type
        if(filepath is not None): # weights from filepath.
            self.load_model(filepath)

    def evaluate(self, input):
        """ Use the network, as policy """
        self.eval() # Turn off training, etc.
        return self(input) # Should use the paren's forward function.

    def store(self,epoch,optimizer,loss, datapath=""): # Need model, and optimizer
        #Store ourself in a file for later use
        save_dir = "models/"+self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir+"/"+ self.name + "_" + str(epoch) # save a new network with an unique ID, name + epoch
        
        # Need to check if directory exists

        torch.save({'epoch':epoch,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':loss,
                    'datapath':datapath,
                    },save_path)

    def load_model(self, path, optimizer=None):
        if os.path.isfile(path): # check if is folder..
            print("loading from ", path)
            #model = model # TheModelClass(*args, **kwargs)
            optimizer = optimizer # TheOptimizerClass(*args, **kwargs)
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            if(optimizer is None):  # Only if we want to keep training.
                return loss, epoch
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return loss, epoch # Return all this info
        else:
            raise ValueError("no checkpoint found at '{}'".format(path))
            #print(" => no checkpoint found at '{}'".format(path))


def train(model, casemanager_train:Datamanager, optimizer, 
        loss_function, batch=10, iterations=1, gpu=False, casemanager_test:Datamanager=None,verbose=1):
    #train_loder is a training row,
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device) # Put model on the specified device. 

    loss_train = 0
    loss_test = 0
    #print("Training network {}".format(model.name))
    if(casemanager_test is None): # * If we want to evaluate against another dataset, different from train.
        # We only train and show loss from this.
        
        for t in range(1,iterations+1): #Itterade dataset x times with batch_size("all") if epochs.
            
            #loss_test = evaluate(casemanager_test, model=model, loss_function=loss_function)
            loss_train_i = train_batch(casemanager_train, 
            model=model, optimizer=optimizer, loss_function=loss_function, batch=batch)
            if(t % verbose == 0 or t == iterations + 1):
                print("itteration {} loss_train: {:.8f}".format(t, loss_train_i))
            loss_train += loss_train_i
        return loss_train/iterations # * average loss
    else:
        for t in range(1,iterations+1): #Itterade dataset x times with batch_size("all") if epochs.
            #loss_test = evaluate(casemanager_test, model=model, loss_function=loss_function)
            loss_train_i = train_batch(casemanager_train, 
            model=model, optimizer=optimizer, loss_function=loss_function, batch=batch)

            loss_test_i = evaluate_test(casemanager_test, batch_size=batch*3,
            model=model, loss_function=loss_function)
            if(t % verbose == 0 or t == iterations + 1):
                print("itteration {}  loss_train: {:.8f} loss_test: {:.8f} ".format(t,loss_train_i, loss_test_i))
            loss_train += loss_train_i ; loss_test += loss_test_i
        return loss_train/iterations, loss_test/iterations # * Return average loss of both.

def weights_init(model): # Will reset states if called again.
    if isinstance(model, nn.Linear):
        init.xavier_uniform_(model.weight) # good init with relus.
        #model.weights.data.fill_(1.0)
        init.constant_(model.bias,0.01) # good init with relus, want every bias to contribute.
        #model.bias.data.zero_() # Bias is set to all zeros.

def train_batch(casemanager:Datamanager, model, optimizer, loss_function, batch):
    # Switch to training mode
    model.train()
    x,y = casemanager.return_batch(batch, model.input_type)# 10 training cases
    y_pred = model(x)
    loss = loss_function(y_pred,y) 
    #print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_test(casemanager:Datamanager, model, loss_function, batch_size):
    # batch_size = "all" is everything 
    model.eval() # Change behaviour of some layers, like no dropout etc.
    with torch.no_grad(): # Turn off gradient calculation requirements, faster.
        data, target = casemanager.return_batch(batch_size, model.input_type) # We might need to resize.
        prediction = model(data)
        return loss_function(prediction,target).item() # Get loss value.

def save_checkpoint(state, filename="models/checkpoint.pth.tar"): #Save as .tar file
    torch.save(state, filename)

def NN_25(name, dim, filepath=None):
    input_dim = 1+(dim*dim) # PID + board
    target_dim = dim*dim
    return Model(
        nn.Linear(input_dim, 25),nn.Tanh(), 
        nn.Linear(25,target_dim), nn.Softmax(dim=-1),
        name=name, 
        filepath=filepath,
        input_type=3
    )