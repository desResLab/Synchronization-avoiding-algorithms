from Training.Model_LSTM import * 
import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import os
from torch.autograd import Variable 
from torch.utils.data import Dataset, DataLoader
from math import ceil
from mpi4py import MPI

# use mpi to save time during hyperparameter tuning
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# in linux, comment this out
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print('\n')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # determine if to use gpu
dpi = 300               # saved image resolution
# the class for the auto batch preparation, referred from the online tutorial
class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] 
        self.x = x
        self.y = y
        # x is usually the training/testing data
        # y is usually the groud truth

    def __len__(self):
        return self.y.shape[0] 
        # number of samples to be batched

    def __getitem__(self, index):
        return self.x[index], self.y[index]        
        # in this case, the first dim of the tensor should be sample label 
        # (one sample is n steps in the past to predict m steps in the future )

# import the whole dataset and shape to the desirable tensor
def Data_extraction_and_sorting(case_num,input_size, n_past, n_future):
    TrainX_total = []
    TrainY_total = []
    for i in range(case_num):
        filename = 'Training/Data-set/Case='+str(i)+'.csv'
        Data_numpy      = genfromtxt(filename, delimiter = ',')

        trainX = [] # the inputs
        trainY = [] # the ground truths

        for i in range(n_past, Data_numpy.shape[0]- n_future + 1):
            trainX.append(Data_numpy[i- n_past:i,:])
            trainY.append(Data_numpy[i:i+n_future,:])

        trainX,trainY = np.array(trainX),np.array(trainY)
        TrainX_total.append(trainX)
        TrainY_total.append(trainY)

    TrainX_total,TrainY_total = np.array(TrainX_total),np.array(TrainY_total)
    #print(TrainX_total.shape,TrainY_total.shape)
    TrainX_total = TrainX_total.reshape((TrainX_total.shape[0]*TrainX_total.shape[1],TrainX_total.shape[2],TrainX_total.shape[3]))
    TrainY_total = TrainY_total.reshape((TrainY_total.shape[0]*TrainY_total.shape[1],TrainY_total.shape[2],TrainY_total.shape[3]))
    return TrainX_total, TrainY_total


# scale all the values in the tensor to [0,1] and return additional max, min values for future inverse scaling
def Scale_to_zero_one(X,Y):
    X_min = X.min()
    X_max = X.max()
    Y_min = Y.min()
    Y_max = Y.max()

    scale_min, scale_max = min(X_min, Y_min), max(X_max, Y_max)

    X = (-X + scale_max) / (-scale_min+scale_max)
    Y = (-Y + scale_max) / (-scale_min+scale_max)
    return X,Y,scale_max, scale_min


#@profile
# training function
def model_training(model, criterion, optimizer, trainloader):
    epo_train_loss = 0 
    model.train()
    for Train_batch, Truth_batch in trainloader:
        out = model(Train_batch)
        loss = criterion(out,Truth_batch)
        epo_train_loss = epo_train_loss + loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return epo_train_loss, model

# testing function
def model_testing(model,criterion,testloader):
    epo_test_loss = 0 
    with torch.no_grad():
        model.eval()
        for Test_batch, Test_Truth_batch in testloader:
            out = model(Test_batch)
            loss = criterion(out,Test_Truth_batch)
            epo_test_loss = epo_test_loss + loss.item()
    return epo_test_loss



# wrapped the main
def main():
    # para spec
    T_portion    = 0.8      # part of the data used for training
    batch_size   = 50       # How many samples used per weights update (a sample is n steps in the past and m steps in the future)
    n_future = 3            # sequence length of the output
    n_past   = 3            # sequence length of the input, this lstm is a seq2seq model
    num_epochs = 3001       # number of epoches
    learning_rate = 1e-4    # initial learning rate for adaptive optimizer and maximum lr for the scheduler
    input_size  = 26*3      # all the dof 
    case_num         = 3    # number of cases used, each case is with a different IC

    # define the model
    hidden_size = int(input_size/4)    # like the hidden layer in MLP
    output_size = input_size           # number of features in the output 
    num_layers = 2                     # number of lstm layer stacked

    model = LSTM_model(input_size,hidden_size,num_layers,output_size)  # our lstm class 
    model = model.to(device)                                           # use GPU instead 
    criterion = torch.nn.MSELoss()                                     # mean-squared error 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimizer to update the model

    # calculate the total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print('total number of trainable parameters is:' + str(total_params))


    # Extract all the data
    X,Y = Data_extraction_and_sorting(case_num, input_size, n_past, n_future )

    # Scale the X,Y into [0,1]
    X,Y,scale_max, scale_min = Scale_to_zero_one(X,Y)


    # start to train the model
    train_loss_save = [] # epoch-wise training loss save
    test_loss_save  = [] # epoch-wise testing loss save
    for epoch in range(num_epochs):
        # to have better generality, we shuffle the training and testin samples for each epoch
        train_length = range(int(len(X)))
        train_slice  = np.random.choice(train_length, size=int(T_portion*len(X)), replace=False) # pick random, un-ordered number from the range of all samples
        test_slice   = np.setdiff1d(train_length, train_slice) # do a set difference to get the test slice

        # Tensorize and move to GPU
        trainX_tensor = Variable(torch.Tensor(X[train_slice,:,:])).to(device) 
        trainY_tensor = Variable(torch.Tensor(Y[train_slice,:,:])).to(device) 
        testX_tensor  = Variable(torch.Tensor(X[test_slice,:,:])).to(device) 
        testY_tensor  = Variable(torch.Tensor(Y[test_slice,:,:])).to(device) 

        
        # use dataloader to do auto-batching
        traindata   =  MyDataset(trainX_tensor, trainY_tensor)
        trainloader =  torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
        testdata    =  MyDataset(testX_tensor, testY_tensor)
        testloader  =  torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True)

        # num of batches for the training set and testing set
        num_batches_train = len(trainloader)
        num_batches_test  = len(testloader)

        # training steps
        epo_train_loss, model = model_training(model, criterion, optimizer, trainloader)
        if epoch%50 == 0 and rank == 0:
            print("Epoch: %d, training loss: %1.7e" % (epoch, epo_train_loss/num_batches_train), 'lr=' + str(optimizer.param_groups[0]['lr']))  
        train_loss_save.append(epo_train_loss/num_batches_train)

        # testing steps
        epo_test_loss = model_testing(model,criterion,testloader)
        if epoch%50 == 0 and rank == 0:
            print("Epoch: %d, testing loss: %1.7e" % (epoch, epo_test_loss/num_batches_test))  
        test_loss_save.append(epo_test_loss/num_batches_test)


    # save the trained model
    PATH = 'Training/Trained_model/'
    os.makedirs(PATH ,exist_ok=True)
    torch.save(model.state_dict(), PATH+'model.pth')

    # save the losses
    np.savetxt(PATH+'Train_loss.csv',train_loss_save,delimiter = ',')
    np.savetxt(PATH+'Test_loss.csv',test_loss_save,delimiter = ',')

    # plot the losses in a semi-logy fashion
    fig = plt.figure(figsize=(22, 8), dpi=dpi)
    plt.figure()
    plt.semilogy(range(num_epochs), train_loss_save,label='training')
    plt.semilogy(range(num_epochs), test_loss_save,label='testing')
    plt.legend()
    fig_name = PATH+'Train-valid-loss.png'
    plt.ylim(1e-11, 1e-1)
    plt.savefig(fig_name,dpi=fig.dpi)

    # plot the losses in a loglog fashion
    fig2 = plt.figure(figsize=(22, 8), dpi=dpi)
    plt.figure()
    plt.loglog(range(num_epochs), train_loss_save,label='training')
    plt.loglog(range(num_epochs), test_loss_save,label='testing')
    plt.legend()
    fig2_name = PATH+'log-log-Train-valid-loss.png'
    plt.ylim(1e-11, 1e-1)
    plt.savefig(fig2_name,dpi=fig2.dpi)


if __name__=="__main__":
    main()