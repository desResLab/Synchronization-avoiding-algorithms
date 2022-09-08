import torch
import torch.nn as nn
import os
from Tools.DNN_tools import *
from Tools.commons import *
from Tools.Distributed_tools import *
import numpy as np
from numpy import genfromtxt
from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # determine if to use gpu

# start main script
for batch_size in [10]:                 # mini batch size (the n_B in paper)
	for learning_rate in [5e-4]:        # initial learning rate (the eta0 in paper)
		for hidden_size in [50]:       # hidden unit size (the n_H in paper)
			
			filter_size = 150                      # sample every xx points from the original data set (the n_s in paper)
			cut_off     = 0.5                      # use xx % of total data (the n_ts in paper)

			# model file name
			Ori_PATH = 'Distributed_save/Rank-'+str(rank)+'/'  # path of the folder
			Save_name = 'nB-'+ str(batch_size) + '-nH-' + str(hidden_size) + '-Lr-' + str(learning_rate) + '-filter=' + str(filter_size)
			PATH = Ori_PATH + Save_name
			os.makedirs(PATH,exist_ok = True)
			if rank == 0:
				print('Current testing:' + Save_name)

			# para spec
			lr_min      = 5e-7              # keep the minimal learning rate the same, avoiding too small update, 
			decay       = 0.998             # learning rate decay rate(the gamma in paper)
			T_portion    = 0.75     		# part of the data used for training, the other for validation
			n_future = 20           		# sequence length of the output (the n_f in paper)
			n_past   = 20           		# sequence length of the input, this lstm is a seq2seq model  (the n_p in paper)      
			num_layers_encoder = 2  		# number of encoder stacked layers
			Bi_dir_encoder     = True 		# if using bidirectional lstm in the encoder
			dp_encoder         = 0.0 		# dropout in encoder, only works for more than two layers
			dp_decoder         = 0.0 		# dropout btw the hidden state and the final dense layer
			training_method      = 'recursive'    # recursively producing the prediction
			#training_method      = 'mtf'         # use mixed teacher's forcing to produce the prediction
			ratio              = 0.6              # part of the prediction by teacher's forcing, if used

			# shared nodes and local nodes information
			shared_path   = 'Results/Shared_Data/Rank='+str(rank)+'_shared.csv'
			Rankwise_path = 'Results/Rankwised_Data/Rank='+str(rank)+'_local_nodes.csv'
			shared_nodes    =  genfromtxt(shared_path, delimiter = ',')
			rankwise_nodes  =  genfromtxt(Rankwise_path,delimiter=',')
			shared_nodes   = shared_nodes.astype(int)
			rankwise_nodes = rankwise_nodes.astype(int)

			# input size of the DNN model
			input_size           = len(shared_nodes)*3

			# construct model
			criterion = nn.MSELoss() # mse loss function
			model = LSTM_encoder_decoder(input_size, hidden_size, num_layers_encoder, Bi_dir_encoder, dp_encoder, dp_decoder)
			model = model.to(device) # send the model to gpu

			# calculate corresponding epoch number
			num_epochs = int(math.log(lr_min/learning_rate, decay))
			if rank == 0:
				print('Number of epoch  is: ' + str(num_epochs))
			# other spec.
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)          # adam optimizer
			lambda1 = lambda epoch: decay ** epoch                                      # lr scheduler by lambda function
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # define lr scheduler with the optim

			# calculate the total number of trainable parameters
			# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
			# print('total number of trainable parameters is:' + str(total_params))
			
			# create data path
			data_path = 'Results/sol_on_shared/rank='+str(rank)+'-shared_dof.hdf5'

			# use cut_off % of data for training (sampled by filter size n_s and later randomly pick 75%)
			X,Y = Dis_data_filtered_subset_coronary(device, input_size, filter_size, n_past, n_future, data_path, cut_off)
			X,Y,scale_max, scale_min = Scale_to_zero_one(X,Y) # feature scaling to -1,0
 			
 			# use all data for testing 
			X_full,Y_full = Dis_data_filtered_subset_coronary(device,input_size, filter_size, n_past, n_future, data_path, 1.0)

			# backup and use the scaling constants during training
			X_backup = scale_forward(X_full, scale_max, scale_min)
			Y_backup = scale_forward(Y_full, scale_max, scale_min)

			# loss placeholders (note: here terms "testing, test" are used, but actually means validation, while some other literature suggests the opposite)
			train_loss_save = [] # epoch-wise training loss save
			test_loss_save  = [] # epoch-wise testing loss save
			train_acc_r2_save = [] # epoch-wise training loss save by r2
			test_acc_r2_save  = [] # epoch-wise testing loss save by r2
			train_acc_rel_save = [] # epoch-wise training loss save by rel
			test_acc_rel_save  = [] # epoch-wise testing loss save by rel

			# find training and validation set
			train_length = range(X.shape[0])
			train_slice  = np.random.choice(train_length, size=int(T_portion*X.shape[0]), replace=False) # pick random, un-ordered number from the range of all samples
			test_slice   = np.setdiff1d(train_length, train_slice) # do a set difference to get the validation slice, this is random but ordered

			# making slices
			trainX_tensor = X[train_slice,:,:]
			trainY_tensor = Y[train_slice,:,:] 

			testX_tensor  = X[test_slice,:,:]
			testY_tensor  = Y[test_slice,:,:]
			
			#------------------------Start the training----------------------------#
			for epoch in range(num_epochs):

				# use dataloader to do auto-batching
				traindata   =  MyDataset(trainX_tensor, trainY_tensor)
				trainloader =  torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True) # shuffle the training mini batches 
				testdata    =  MyDataset(testX_tensor, testY_tensor)
				testloader  =  torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False) # dont shuffle the validation mini batches
				num_batches_train = len(trainloader) # total # of training batches
				num_batches_test  = len(testloader) # total # of validation batches

				# Train steps
				epo_train_loss, epo_train_acc_r2, epo_train_acc_rel, model = model_train(device, model,trainloader, criterion, optimizer,
									n_future, training_method = training_method, ratio = ratio)
				if rank == 0 and epoch%50 == 0:
					print("Epoch: %d, mse training loss: %1.5e" % (epoch, epo_train_loss/num_batches_train) , ", R2 accuracy: %.3f" % (epo_train_acc_r2/num_batches_train) ,  'lr=' + str(optimizer.param_groups[0]['lr']))  
				train_loss_save.append(epo_train_loss/num_batches_train)
				train_acc_r2_save.append(epo_train_acc_r2/num_batches_train)
				train_acc_rel_save.append(epo_train_acc_rel/num_batches_train)

				# validation steps
				epo_test_loss, epo_test_acc_r2, epo_test_acc_rel  = model_test(device, model, testloader, criterion, n_future)
				if rank == 0  and epoch%50 == 0 :
					print("Epoch: %d, mse testing loss: %1.5e" % (epoch, epo_test_loss/num_batches_test), ", R2 accuracy: %.3f" % (epo_test_acc_r2/num_batches_test))  
				test_loss_save.append(epo_test_loss/num_batches_test)
				test_acc_r2_save.append(epo_test_acc_r2/num_batches_test)
				test_acc_rel_save.append(epo_test_acc_rel/num_batches_test)

				scheduler.step() # update learning rate


			# plot train and validation curves and save them
			fig1 = plt.figure(figsize=(16, 8))
			plt.subplot(1,2,1)
			plt.semilogy(train_loss_save, label = 'train')
			plt.semilogy(test_loss_save, label = 'test')
			plt.xlabel('epoch')
			plt.ylim([1e-10, 1e-1])
			plt.legend()

			plt.subplot(1,2,2)
			plt.plot(train_acc_r2_save, label = 'tran:R2')
			plt.plot(test_acc_r2_save, label = 'test:R2')
			plt.plot(train_acc_rel_save, label = 'train:Rel')
			plt.plot(test_acc_rel_save, label = 'test:Rel')
			plt.xlabel('epoch')
			plt.ylim([0.9, 1])
			plt.legend()
			fig_name = PATH + '/train-test-loss-acc.png'
			plt.savefig(fig_name)
			
			train_save_name = PATH + '/train_loss.csv'
			test_save_name  = PATH + '/test_loss.csv'
			np.savetxt(train_save_name, train_loss_save, delimiter = ',')
			np.savetxt(test_save_name, test_loss_save,   delimiter = ',')

			train_acc_r2_savename = PATH + '/train_acc_r2.csv'
			test_acc_r2_savename  = PATH + '/test_acc_r2.csv'
			np.savetxt(train_acc_r2_savename, train_acc_r2_save, delimiter = ',')
			np.savetxt(test_acc_r2_savename, test_acc_r2_save,   delimiter = ',')

			train_acc_rel_savename = PATH + '/train_acc_rel.csv'
			test_acc_rel_savename  = PATH + '/test_acc_rel.csv'
			np.savetxt(train_acc_rel_savename, train_acc_rel_save, delimiter = ',')
			np.savetxt(test_acc_rel_savename, test_acc_rel_save,   delimiter = ',')


			# save the trained model
			model_save_name   = PATH + '/model.pth'
			torch.save(model.state_dict(), model_save_name)

			