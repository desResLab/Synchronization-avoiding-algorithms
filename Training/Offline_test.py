from Model_LSTM import * 
import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt


# in linux, comment this out
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


test_case = 4           # could try a case NOT seen by the training
start     = 12000       # Where to start the prediction, ramped period does not matter
n_future = 3            # sequence length of the output
n_past   = 3            # sequence length of the input, this lstm is a seq2seq model
input_size  = 26*3      # all the dof 


# define the model
hidden_size = int(input_size/4)    # like the hidden layer in MLP
output_size = input_size           # number of features in the output 
num_layers = 1                     # number of lstm layer stacked

model = LSTM_model(input_size,hidden_size,num_layers,output_size)  # our lstm class 
# load the learned model weights
model.load_state_dict(torch.load('Case3-Trained_model/model.pth',map_location=torch.device('cpu')))

# load the data file
filename       = 'Data-set/Case='+str(test_case)+'.csv'
Data_numpy      = genfromtxt(filename, delimiter = ',')


# scale the data
scale_min = Data_numpy.min()
scale_max = Data_numpy.max()

Data_numpy = (-Data_numpy + scale_max) / (-scale_min+scale_max)


# start to gather model input
input_batch       = torch.zeros((1, n_past, input_size))
truth_batch       = np.zeros((1, n_future, input_size))

# insert the data
input_batch[0,:,:] = torch.from_numpy(Data_numpy[start-n_past : start,:])
model.eval()

N = 51  # number of model usage, number of predicted steps = N*n_future
for i in range(N):
	prediction = model(input_batch)
	input_batch = prediction

prediction = prediction.detach().numpy()
truth_batch[0,:,:] = Data_numpy[start+n_past*(N-1) : start+n_past*(N-1)+n_future,:]

# plot scaled dx
plt.figure(figsize=(22, 8))
plt.subplot(1,3,1)
plt.plot(prediction[0,0,0::3], label='predicted')
plt.plot(truth_batch[0,0,0::3],'--', label='truth')
plt.ylim([0,1])
plt.subplot(1,3,2)
plt.plot(prediction[0,0,1::3])
plt.plot(truth_batch[0,0,1::3],'--')
plt.ylim([0,1])
plt.subplot(1,3,3)
plt.plot(prediction[0,0,2::3])
plt.plot(truth_batch[0,0,2::3],'--')
plt.ylim([0,1])

# plot scaled dy
plt.figure(figsize=(22, 8))
plt.subplot(1,3,1)
plt.plot(prediction[0,1,0::3])
plt.plot(truth_batch[0,1,0::3],'--')
plt.ylim([0,1])
plt.subplot(1,3,2)
plt.plot(prediction[0,1,1::3])
plt.plot(truth_batch[0,1,1::3],'--')
plt.ylim([0,1])
plt.subplot(1,3,3)
plt.plot(prediction[0,1,2::3])
plt.plot(truth_batch[0,1,2::3],'--')
plt.ylim([0,1])

# plot scaled dz
plt.figure(figsize=(22, 8))
plt.subplot(1,3,1)
plt.plot(prediction[0,2,0::3])
plt.plot(truth_batch[0,2,0::3],'--')
plt.ylim([0,1])
plt.subplot(1,3,2)
plt.plot(prediction[0,2,1::3])
plt.plot(truth_batch[0,2,1::3],'--')
plt.ylim([0,1])
plt.subplot(1,3,3)
plt.plot(prediction[0,2,2::3])
plt.plot(truth_batch[0,2,2::3],'--')
plt.ylim([0,1])

plt.show()