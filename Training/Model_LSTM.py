import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# implement a vanilla lstm model
class LSTM_model(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,output_size):
		super(LSTM_model,self).__init__()

		self.input_size  = input_size  # input size
		self.hidden_size = hidden_size # hidden state
		self.num_layers  = num_layers  # number of rnn layers
		self.output_size = output_size # output size 

		self.my_LSTM = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True) # default RNN 
		self.fc     =  nn.Linear(self.hidden_size, self.output_size) # fully connected last layer

	def forward(self,x):
		h0  = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # hidden state of lstm
		c0  = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # cell state of lstm

		out, (hn,cn) = self.my_LSTM(x, (h0,c0))

		return self.fc(out)