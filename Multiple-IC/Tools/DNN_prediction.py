#from Offline_training.NN_design import *
from Dis_training.Single_step_ResNet import *
#from Offline_training_Dis.ResNet_Subset import *
import numpy as np
from numpy import genfromtxt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools.commons import *
from Tools.Distributed_tools import *

# def Res_predicting(rank,size,start):
# 	local_shared_name = 'Results/Shared_Data/Rank='+str(rank)+'_shared.csv' # local shared nodes
# 	local_shared = genfromtxt(local_shared_name,delimiter=',')

# 	local_predication = np.zeros( (len(local_shared)*3,1)) 


# 	# Define pre-training parameters
# 	num_input  = len(local_shared)*3
# 	num_output = len(local_shared)*3
# 	num_neu    = int(num_output)

# 	# load all shared data
# 	filename = 'Results/Res_Net_data/S/Step_'+str(start-1)+'_Rank='+str(rank)+'_Acc.csv'
# 	local_syn_numpy  = genfromtxt(filename, delimiter = ',')

		
# 	local_syn_tensor = torch.from_numpy(local_syn_numpy)

# 	# load learned data
# 	data_save = 'Offline_training/acc/Trained_model_no_norm_dropout/Rank'+str(rank)+'-learned-acc.pth'
# 	model = ResNet_block(num_input,num_neu,num_output)

# 	# load the weights
# 	model.load_state_dict(torch.load(data_save))
# 	model.eval()
# 	# use trained model to predict
# 	Prediction = model( np.transpose(local_syn_tensor.float()) ) 

# 	return (Prediction.detach().numpy()).reshape(len(Prediction),1) 



# def Res_predicting_inc(rank,size,start):
# 	local_shared_name = 'Results/Shared_Data/Rank='+str(rank)+'_shared.csv' # local shared nodes
# 	local_shared = genfromtxt(local_shared_name,delimiter=',')

# 	local_predication = np.zeros( (len(local_shared)*3,1)) 


# 	# Define pre-training parameters
# 	num_input  = len(local_shared)*3
# 	num_output = len(local_shared)*3
# 	num_neu    = int(num_output)

# 	# load all shared data
# 	filename1 = 'Results/Res_Net_data/S/Step_'+str(start-1)+'_Rank='+str(rank)+'_Acc.csv'
# 	local_syn_numpy1  = genfromtxt(filename1, delimiter = ',')

# 	filename2 = 'Results/Res_Net_data/S/Step_'+str(start-2)+'_Rank='+str(rank)+'_Acc.csv'
# 	local_syn_numpy2  = genfromtxt(filename2, delimiter = ',')

# 	local_syn_numpy   = local_syn_numpy1 - local_syn_numpy2  # the increment
		
# 	local_syn_tensor = torch.from_numpy(local_syn_numpy)

# 	# load learned data
# 	data_save = 'Offline_training/acc-inc/Trained_model_no_norm_dropout/Rank'+str(rank)+'-learned-acc.pth'
# 	model = ResNet_block(num_input,num_neu,num_output)

# 	# load the weights
# 	model.load_state_dict(torch.load(data_save))
# 	model.eval()
# 	# use trained model to predict
# 	Prediction = model( np.transpose(local_syn_tensor.float()) ) 


# 	return (Prediction.detach().numpy() + local_syn_numpy1).reshape(len(Prediction),1) 

# Displacement DNN prediction by the entire dataset, serial code
def Dis_prediction(width,d0, Dirichlet):
	d0 = d0.reshape((width*3))
	d0 = torch.from_numpy(d0)
	# Define pre-training parameters
	num_input  = width*3 
	num_output = width*3 
	num_neu    = width*3 * 2

	data_save = 'Tools/Trained_model_12000_complex/Trained-time.pth'

	model = ResNet_block_2steps(num_input,num_neu,num_output)

	# load the weights
	model.load_state_dict(torch.load(data_save))
	model.eval()
	# use trained model to predict
	Prediction = model( np.transpose(d0.float()) ) 
	
	d1 =  Prediction.detach().numpy()

	# strong enforce the dirichlet bc to the predicted solution
	for i in range(len(d1)):
		if i in Dirichlet:
			d1[i] = 0 

	return d1




# Displacement DNN prediction by the subset of the dataset, parallel code

def DNN_prediction_dis_subset(rank,start):
	nM            = 9        # number of previous steps used, the memory


	# shared nodes information
	local_shared_name = 'Results/Shared_Data/Rank='+str(rank)+'_shared.csv' # local shared nodes
	local_shared      = genfromtxt(local_shared_name,delimiter=',')
	Dataset_numpy    = np.zeros( (len(local_shared)*3,nM) )     # placeholder for the input

	# load the model
	data_save = 'Offline_training_Dis/Trained_model/Trained-subset-rank='+str(rank)+'.pth'

	# Define pre-training parameters
	num_input  = len(local_shared)*3*nM
	num_output = len(local_shared)*3
	num_neu    = num_output*2
	model      = ResNet_block_subset(num_input,num_neu,num_output)
	# load the weights
	model.load_state_dict(torch.load(data_save))
	model.eval()


	# read from the files, the memory
	for i in range(nM):
		syn_filename = 'Results/Res_Net_data/S/Step_'+str(start-nM+i)+'_Rank='+str(rank)+'_Dis.csv' # the training data
		syn_dis      = genfromtxt(syn_filename, delimiter = ',')
		Dataset_numpy[:,i] = syn_dis

	sorted_slices = np.reshape(Dataset_numpy,(len(local_shared)*3*nM,1),order='F') # reshape for compatible with identity map
	test_vector = torch.from_numpy(sorted_slices)
	Acc_val = model( np.transpose(test_vector.float()),len(local_shared)*3)
	Acc_pred = np.transpose( (Acc_val.detach().numpy())[0])

	return Acc_pred



# function to read learned data from file and update the value on shared nodes
def Load_DNN_data_dis(d1, rank, Local_nodes,shared_nodes,start):
	learned_dis = DNN_prediction_dis_subset(rank,start)
	local_shared_dof = node_to_dof(3, [0,1,2], local_mat_node(shared_nodes,Local_nodes))
	learned_dis = learned_dis.reshape( (len(learned_dis),1) )
	d1[local_shared_dof] = learned_dis
	return d1