import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# an overkilled case of 4 blocks

class ResNet_block_4steps(nn.Module):
	def __init__(self, Number_of_input, Number_of_neurons, Number_of_output):   # constructor
		super().__init__()               # allow inherience from nn.Module

		self.NI  = Number_of_input       # number of input neurons
		self.NN  = Number_of_neurons     # number of hidden layer neurons
		self.NO  = Number_of_output      # number of output neurons

		#self.act   = nn.Tanh() # so as Xiu et.al 2019
		self.act   = nn.ReLU()
		#self.act    = nn.Sigmoid() 

		self.dropout_inlet  = nn.Dropout(0.0)
		self.dropout_hidden = nn.Dropout(0.0)
		self.dropout_outlet = nn.Dropout(0.0)

		self.MLP = nn.Sequential(
						nn.Linear(self.NI, self.NN),
						self.dropout_inlet,
						self.act,
						nn.Linear(self.NN, self.NN),
						self.dropout_hidden,
						self.act,
						nn.Linear(self.NN, self.NI),
						self.dropout_hidden,
						self.act,
						)


		self.Outlet_layer = nn.Sequential(  self.dropout_outlet,
											nn.Linear(self.NI, self.NO),
											)	
	# only one residual-block here
	def forward(self,x):
		identity = x
		x1 = self.MLP(x) + identity    # apply residual 

		identity2 = x1
		x2        = self.MLP(x1) + identity2

		identity3 = x2
		x3        = self.MLP(x2) + identity3

		identity4 = x3
		x4        = self.MLP(x3) + identity4

		return self.Outlet_layer(x4)