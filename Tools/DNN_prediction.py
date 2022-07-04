from Tools.Distributed_tools import *
from Tools.Dynamic_solver import *
from Tools.commons import *
from Tools.DNN_tools import *
import torch
import torch.nn as nn
import meshio
import numpy as np
from numpy import genfromtxt
import os
from torch.autograd import Variable 
from mpi4py import MPI
import h5py
import meshio

# call pre-trained model
def call_model(device, filter_size, input_size, hidden_size, model_path):

	# para spec
	num_layers_encoder = 2      # number of encoder stacked layers
	Bi_dir_encoder     = True   # if using bidirectional lstm in the encoder
	dp_encoder         = 0.0    # dropout in encoder, only works for more than two layers
	dp_decoder         = 0.0    # dropout btw the hidden state and the final dense layer

	model = LSTM_encoder_decoder(input_size, hidden_size, num_layers_encoder, Bi_dir_encoder, dp_encoder, dp_decoder)
	model.load_state_dict(torch.load(model_path,map_location=device)) # load learned weights
	model = model.to(device) 
	return model


# use trained model to predict
def encoder_decoder_predictor(device, n, model, n_p, n_f, n_s, input_size, d_sol, scale_max, scale_min):
	NF = np.zeros((n_s*n_f,input_size))
	for i in range(n_s):
		Npi = np.arange(i+n-n_p*n_s,i+n-1,n_s)
		Nfi = np.arange(i+n, n + i+ n_f*n_s-1,n_s)

		X_input = d_sol[Npi,:]
		X_input = scale_forward(X_input,scale_max, scale_min)
		X_input = torch.from_numpy(X_input).float().to(device)
		X_output = model_predict(device, model, X_input, n_f)
		X_output = scale_it_back(X_output,scale_max,scale_min)

		shifted_array = Nfi - n
		NF[shifted_array,:] = X_output.cpu().detach().numpy()
	return NF 




























