# neural network training
import torch
import torch.nn as nn
import os
from Tools.DNN_tools import *
from Tools.commons import *
from Tools.Distributed_tools import *
from Tools.plotting_tools import *
import numpy as np
from numpy import genfromtxt
from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # determine if to use gpu

# start main script
for batch_size in [50,20,10,5]:                 # search batch size
	for learning_rate in [5e-5,5e-4,5e-3]:   # search learning rate
		for hidden_size in [20, 50, 100]:    # search hidden unit size
			
			filter_size = 80                      # sample every xx points from the original data set
			cut_off     = 0.5                     # use xx % of total data

			Ori_PATH = 'Distributed_save/Rank-'+str(rank)+'/'  # path of the folder
			Save_name = 'nB-'+ str(batch_size) + '-nH-' + str(hidden_size) + '-Lr-' + str(learning_rate) + '-filter=' + str(filter_size)
			PATH = Ori_PATH + Save_name
			os.makedirs(PATH,exist_ok = True)
			if rank == 0:
				print('Current testing:' + Save_name)

			# para spec
			lr_min      = 5e-7              # keep the minimal learning rate the same, avoiding too small update
			decay       = 0.9995            # learning rate decay rate
			T_portion    = 0.75     		# part of the data used for training
			n_future = 20           		# sequence length of the output
			n_past   = 20           		# sequence length of the input, this lstm is a seq2seq model        
			num_layers_encoder = 2  		# number of encoder stacked layers
			Bi_dir_encoder     = True 		# if using bidirectional lstm in the encoder
			dp_encoder         = 0.0 		# dropout in encoder, only works for more than two layers
			dp_decoder         = 0.0 		# dropout btw the hidden state and the final dense layer
			training_method      = 'recursive'    # recursively producing the prediction
			#training_method      = 'mtf'         # use mixed teacher's forcing to produce the prediction
			ratio              = 0.6              # part of the prediction by teacher's forcing

			# shared nodes information
			shared_path   = 'Results/Shared_Data/Rank='+str(rank)+'_shared.csv'
			Rankwise_path = 'Results/Rankwised_Data/Rank='+str(rank)+'_local_nodes.csv'

			shared_nodes    =  genfromtxt(shared_path, delimiter = ',')
			rankwise_nodes  =  genfromtxt(Rankwise_path,delimiter=',')

			shared_nodes   = shared_nodes.astype(int)
			rankwise_nodes = rankwise_nodes.astype(int)

			input_size           = len(shared_nodes)*3

			# construct model
			criterion = nn.MSELoss()
			model = LSTM_encoder_decoder(input_size, hidden_size, num_layers_encoder, Bi_dir_encoder, dp_encoder, dp_decoder)
			model = model.to(device) 

			# calculate corresponding epoch number
			num_epochs = int(math.log(lr_min/learning_rate, decay))
			if rank == 0:
				print('Number of epoch  is: ' + str(num_epochs))

			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
			lambda1 = lambda epoch: decay ** epoch                                      # lr scheduler by lambda function
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # define lr scheduler with the optim

			# calculate the total number of trainable parameters
			# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
			# print('total number of trainable parameters is:' + str(total_params))
			
			data_path = 'Results/sol_on_shared/rank='+str(rank)+'-shared_dof.hdf5'

			# use real disaplcement data, full cantilever
			X,Y = Dis_data_filtered_subset_coronary(device, input_size, filter_size, n_past, n_future, data_path, cut_off)
			X,Y,scale_max, scale_min = Scale_to_zero_one(X,Y)

			# Y need more data 
			X_full,Y_full = Dis_data_filtered_subset_coronary(device,input_size, filter_size, n_past, n_future, data_path, 1.0)

			# back up for later forcasting
			X_backup = scale_forward(X_full, scale_max, scale_min)
			Y_backup = scale_forward(Y_full, scale_max, scale_min)


			#------------------------Start the training----------------------------#
			train_loss_save = [] # epoch-wise training loss save
			test_loss_save  = [] # epoch-wise testing loss save
			train_acc_r2_save = [] # epoch-wise training loss save by r2
			test_acc_r2_save  = [] # epoch-wise testing loss save by r2
			train_acc_rel_save = [] # epoch-wise training loss save by rel
			test_acc_rel_save  = [] # epoch-wise testing loss save by rel

			for epoch in range(num_epochs):

				# to have better generality, we shuffle the training/testing samples for each epoch
				train_length = range(X.shape[0])
				train_slice  = np.random.choice(train_length, size=int(T_portion*X.shape[0]), replace=False) # pick random, un-ordered number from the range of all samples
				test_slice   = np.setdiff1d(train_length, train_slice) # do a set difference to get the test slice, this is random but ordered

				# making slices
				trainX_tensor = X[train_slice,:,:]
				trainY_tensor = Y[train_slice,:,:] 

				testX_tensor  = X[test_slice,:,:]
				testY_tensor  = Y[test_slice,:,:]

				# use dataloader to do auto-batching
				traindata   =  MyDataset(trainX_tensor, trainY_tensor)
				trainloader =  torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
				testdata    =  MyDataset(testX_tensor, testY_tensor)
				testloader  =  torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True)
				num_batches_train = len(trainloader)
				num_batches_test  = len(testloader)

				# Note: we do random slicing for the data samples and random batching in data-loader
				# this seems a little bit redundant but no other good ways can be thought of now. But as
				# long as they are random, it's probably fine.

				# Train steps
				epo_train_loss, epo_train_acc_r2, epo_train_acc_rel, model = model_train(device, model,trainloader, criterion, optimizer,
									n_future, training_method = training_method, ratio = ratio)
				if rank == 0 and epoch%50 == 0:
					print("Epoch: %d, mse training loss: %1.5e" % (epoch, epo_train_loss/num_batches_train) , ", R2 accuracy: %.3f" % (epo_train_acc_r2/num_batches_train) ,  'lr=' + str(optimizer.param_groups[0]['lr']))  
				train_loss_save.append(epo_train_loss/num_batches_train)
				train_acc_r2_save.append(epo_train_acc_r2/num_batches_train)
				train_acc_rel_save.append(epo_train_acc_rel/num_batches_train)

				# Test steps
				epo_test_loss, epo_test_acc_r2, epo_test_acc_rel  = model_test(device, model, testloader, criterion, n_future)
				if rank == 0  and epoch%50 == 0 :
					print("Epoch: %d, mse testing loss: %1.5e" % (epoch, epo_test_loss/num_batches_test), ", R2 accuracy: %.3f" % (epo_test_acc_r2/num_batches_test))  
				test_loss_save.append(epo_test_loss/num_batches_test)
				test_acc_r2_save.append(epo_test_acc_r2/num_batches_test)
				test_acc_rel_save.append(epo_test_acc_rel/num_batches_test)

				scheduler.step()



			# plot train and test curves
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


			# save the model
			model_save_name   = PATH + '/model.pth'
			torch.save(model.state_dict(), model_save_name)

			# use the trained model to predict    
			N     = 60    # number of model usage, for each usage, we have nf predictions, sequentially
			start = 0     # starting point

			test_batch = X_backup[start,:, :]

			Y_plot    = np.zeros((N*n_future,input_size)) # save truth
			Pred_plot = np.zeros((N*n_future,input_size)) # save prediction

			for i in range(N):
				backup = test_batch.clone() # use clone to avoid in-place operation
				truth_batch = Y_backup[start + n_future*(i),:, :] # the truth
				Y_plot[n_future*i:n_future*(i+1),:]    = truth_batch[:,:].cpu().detach().numpy()
				outputs = model_predict(device,model, test_batch , n_future) # the modeled solution
				# shift the test batch to use previous predictions
				test_batch[:n_past-n_future,:] =  backup[n_future:,:] # shift upwards
				test_batch[n_past-n_future:,:] = outputs              # use predictions
				Pred_plot[n_future*i:n_future*(i+1),:] = outputs[:,:].cpu().detach().numpy()

			# rescaling
			Y_plot    = scale_it_back(Y_plot, scale_max.item(),scale_min.item())
			Pred_plot = scale_it_back(Pred_plot, scale_max.item(), scale_min.item())


			# calculate l2 norm
			Diff = Y_plot - Pred_plot
			Y_plot_norm = np.sqrt(np.sum(Y_plot**2,axis=1)).reshape((len(Y_plot),1))
			e_l2        = np.sqrt(np.sum(Diff**2,axis=1)).reshape((len(Diff),1))
			e_l2_rel    = e_l2/Y_plot_norm
			error_save_name   = PATH + '/error_time.csv'
			np.savetxt(error_save_name, np.concatenate([e_l2, e_l2_rel], axis=1), delimiter = ',')

			E_l2  = np.mean(e_l2)/input_size
			E_l2_rel = np.mean(e_l2_rel)/input_size
			Error_save_name   = PATH + '/error_total.csv'
			np.savetxt(Error_save_name, np.array([E_l2, E_l2_rel]), delimiter = ',')

			# plot the predictions
			mk = 8
			plot_subset(Y_plot, Pred_plot, mk, Save_name, PATH)