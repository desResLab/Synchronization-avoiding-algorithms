import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from Single_step_ResNet import *

#hyperparameters
# epoch: 300~600
# (initial) learning rate: 0.01 ~0.05
# mini-B: 10~20
# total_pair: 20000+


dpi = 200
Nepoch    = 500             # number of epochs
learning_rate =  .01        # initial learning rate, will be tuned by scheduler 
mini_B        =  20         # mini batch size
case_num      = 5
case_item_num = 12000
total_pair    = int(case_num*case_item_num)             # total number of data pairs, i.e. we have total_pair+1 data set
B_num         = int(total_pair/mini_B)                  # num of mini batches
T_portion     = 0.8       
width         = 26

# gather data
Training_numpy = np.zeros((width*3,total_pair))
Testing_numpy  = np.zeros((width*3,total_pair))

for i in range(case_num):
	for j in range(case_item_num):
		counter = i*case_item_num + j
		filename1 = 'Data-set/cases='+str(i)+'/num='+str(j)+'.csv'
		Dis1      = genfromtxt(filename1, delimiter = ',')
		Dis1      = Dis1.reshape((width*3))
		Training_numpy[:,counter]    =  Dis1
		filename2 = 'Data-set/cases='+str(i)+'/num='+str(j+1)+'.csv'
		Dis2      = genfromtxt(filename2, delimiter = ',')
		Dis2      = Dis2.reshape((width*3))
		Testing_numpy[:,counter]     =  Dis2


# Define pre-training parameters
num_input  = width*3 
num_output = width*3 
num_neu    = width*3 * 2

model   = ResNet_block_2steps(num_input,num_neu,num_output)
loss_fn = torch.nn.MSELoss()                  # loss function, default is mean

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lambda1 = lambda epoch: 0.98 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# split into batches
training_size   = int(B_num * T_portion)
validation_size = int(B_num - training_size)

# randomize it and store the list 
target_bts = range(total_pair)
training_bts    = np.zeros((training_size,mini_B))
for i in range(training_size):
	training_bts[i,:]   = np.random.choice(target_bts, size=mini_B, replace=False)
	target_bts  = np.setdiff1d(target_bts, training_bts[i,:])


training_loss   = []
validation_loss = []

train_accuracy = []
validation_accuracy = []

# start off the training
for epoch in range(1, Nepoch + 1):
	epo_t_loss = 0
	epo_t_acc  = 0

	epo_v_loss = 0
	epo_v_acc  = 0

	 # train
	for Bt1 in range(len(training_bts[:,1])):   # loop over each batches, i.e rows of target_list
		model.train()
		optimizer.zero_grad()
		loss_t = 0
		acc_t  = 0
		
		train_list = [int(label) for label in training_bts[Bt1,:]]
		# locate the training data and its corresponding true values
		mini_training = torch.from_numpy(Training_numpy[:,train_list])
		mini_testing  = torch.from_numpy(Testing_numpy[:,train_list])

		# need transpose here since our columns are corresponding to each batch
		Acc_pre = model(np.transpose(mini_training.float()))
		loss_t  = loss_fn(Acc_pre,  np.transpose(mini_testing.float()))

		error_norm  =  np.linalg.norm( (np.transpose(mini_testing)) - Acc_pre.detach().numpy() )
		acc_t       =  1- error_norm/np.linalg.norm( (np.transpose(mini_testing).float()).numpy() )

		epo_t_loss += loss_t.detach()
		epo_t_acc  += acc_t

		loss_t.backward()
		optimizer.step()


	print('current epoch is:' + str(epoch) + ',training loss is:' \
						+str(epo_t_loss/training_size)+ ',training accuracy is:' +str(epo_t_acc/training_size) \
						+ ', lr=' + str(optimizer.param_groups[0]['lr']))
	
	training_loss.append(epo_t_loss/training_size)
	train_accuracy.append(epo_t_acc/training_size)


	# validate
	# Note: in this stage, we mimic the actual forcasting stage, i.e.: no more mini-batches
	model.eval() 
	with torch.no_grad():      # no grad computation
		for val in target_bts: # the remaining pairs not used in the training stage
			validation_vector = torch.from_numpy(Training_numpy[:,int(val)])
			validation_vector_true = torch.from_numpy(Testing_numpy[:,int(val)])

			Acc_val = model( np.transpose(validation_vector.float()))		
			loss_v  = loss_fn(Acc_val,  (np.transpose(validation_vector_true)).float())

			error_norm  = np.linalg.norm( (np.transpose(validation_vector_true)) - Acc_val.detach().numpy() )
			acc_v   =  1- error_norm/np.linalg.norm( (np.transpose(validation_vector_true)).float().numpy() )
		
			epo_v_loss += loss_v.detach()
			epo_v_acc  += acc_v


	print('current epoch is:' + str(epoch) + ',validation loss is:' \
						 +str(epo_v_loss/len(target_bts))+ ',validation accuracy is:' +str(epo_v_acc/len(target_bts)))

	validation_loss.append(epo_v_loss/len(target_bts))
	validation_accuracy.append(epo_v_acc/len(target_bts))

	# reshuffling
	target_bts = range(total_pair)
	training_bts    = np.zeros((training_size,mini_B))
	for i in range(training_size):
		training_bts[i,:]   = np.random.choice(target_bts, size=mini_B, replace=False)
		target_bts  = np.setdiff1d(target_bts, training_bts[i,:])

	scheduler.step()  # update lr 



# plot the loss and accuracy
fig = plt.figure(figsize=(22, 8), dpi=dpi)
plt.figure()
plt.semilogy(range(Nepoch),training_loss,label='training')
plt.semilogy(range(Nepoch), validation_loss,label='validation')
plt.legend()
fig_name = 'Train-valid-loss.png'
#plt.ylim(1e-10, 1e-3)
plt.savefig(fig_name,dpi=fig.dpi)

fig = plt.figure(figsize=(22, 8), dpi=dpi)
plt.figure()
plt.plot(range(Nepoch),train_accuracy,label='training')
plt.plot(range(Nepoch), validation_accuracy,label='validation')
plt.legend()
fig_name = 'Train-valid-accuracy.png'
plt.ylim(0, 1.1)
plt.savefig(fig_name,dpi=fig.dpi)


# save the trained weights
data_save = 'Trained_model_12000_complex/Trained-time.pth'
torch.save(model.state_dict(), data_save)
