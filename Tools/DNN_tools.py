# LSTM-encoder-decoder reference: https://github.com/lkulowski/LSTM_encoder_decoder
import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import os
from torch.autograd import Variable 
from torch.utils.data import Dataset, DataLoader
import h5py

# the lstm encoder-decoder models, code structure referred from: https://github.com/lkulowski/LSTM_encoder_decoder
# The encoder
class LSTM_Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, Bi_dir, dp):
    super(LSTM_Encoder,self).__init__()
    
    self.input_size  = input_size          # number of input features
    self.hidden_size = hidden_size         # number of hidden features
    self.num_layers  = num_layers          # number of stacked lstm layers
    self.Bi_dir      = Bi_dir              # Bidirection?
    self.dp          = dp                  # Dropout rate, note that if dropout to be used, we need at least two layers

    if self.Bi_dir == True:
      self.D = 2      # bidirectional lstm
    else:
      self.D = 1      # unidirectional lstm

    # call build-in lstm function
    self.lstm_encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, batch_first=True, 
                                dropout=self.dp, bidirectional=self.Bi_dir)

  # every forward, we just need the final hidden state
  def forward(self,x):
    _, (hn,cn) = self.lstm_encoder(x) # x of shape: N X L X H_input
    # hn,cn of shape: D∗num_layers x N x H_hidden 
    hn = hn.view(self.num_layers, self.D, x.shape[0], self.hidden_size)
    cn = cn.view(self.num_layers, self.D, x.shape[0], self.hidden_size)
    # only use the last layer
    hn_output = hn[-1]
    cn_output = cn[-1]
    
    ##########----------output-----------------------########### ref:https://discuss.pytorch.org/t/bidirectional-3-layer-lstm-hidden-output/41336/2
    # output should be size of N x H_hidden*D
    if self.D == 2:
      # deal with the two directions
      hn_1, hn_2 = hn_output[0], hn_output[1]           # two directions
      hn_final = torch.cat((hn_1, hn_2), 1)             # Concatenate both states

      cn_1, cn_2 = cn_output[0], cn_output[1]           # two directions
      cn_final = torch.cat((cn_1, cn_2), 1)             # Concatenate both states
      return hn_final.unsqueeze(0), cn_final.unsqueeze(0)
    else:
      return hn_output, cn_output




# The decoder
# making no sense to put a bidirectional layer here? since it will be fed a single input and recursively produce the output
# we also dont give stacked layer here
class LSTM_Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, Bi_dir, dp):
    super(LSTM_Decoder,self).__init__()

    self.input_size  = input_size
    if Bi_dir == True:
      self.hidden_size = hidden_size*2 # bidirection, then double the hidden size
    else:
      self.hidden_size = hidden_size
    self.dp = dp
    self.lstm_decoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
    self.fc = nn.Linear(self.hidden_size, input_size) #outputsize is the same as the input size
    self.dropout = nn.Dropout(self.dp)
  
  def forward(self,x,encoded_hn, encoded_cn):
    #print(x.shape, encoded_hn.shape, encoded_cn.shape)
    out, (hn_decoded, cn_decoded) = self.lstm_decoder(x.unsqueeze(1), (encoded_hn, encoded_cn)) # x should be of the size: N X 1 X H_input, 1 because we do it recursively
    return self.fc(self.dropout(out.squeeze(1))), hn_decoded, cn_decoded



# integrated seq2seq model
class LSTM_encoder_decoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers_encoder, Bi_dir_encoder, dp_encoder, dp_decoder):
    super(LSTM_encoder_decoder, self).__init__()

    self.input_size  = input_size
    self.hidden_size = hidden_size
    self.num_layers_encoder =  num_layers_encoder
    self.Bi_dir_encoder = Bi_dir_encoder
    self.dp_encoder = dp_encoder
    self.dp_decoder = dp_decoder


    self.encoder = LSTM_Encoder( self.input_size, self.hidden_size, self.num_layers_encoder, self.Bi_dir_encoder, self.dp_encoder)
    self.decoder = LSTM_Decoder( self.input_size, self.hidden_size, self.Bi_dir_encoder, self.dp_decoder)



# train the model
def model_train(device, model,trainloader, criterion,  optimizer,  n_future, training_method = 'recursive', ratio = 0.5):
  epo_train_loss = 0
  epo_train_acc_rel  = 0
  epo_train_acc_r2   = 0

  model.train()

  for Train_batch, Truth_batch in trainloader:
      
      # Zero-out the gradient
      optimizer.zero_grad()
      
      # call encoder
      e_hn,e_cn = model.encoder(Train_batch)
      
      decoder_input  = Train_batch[:,-1,:]  
      decoder_hidden = e_hn
      decoder_cell   = e_cn
    
      # recursively decoding
      outputs = torch.zeros((Train_batch.shape[0], n_future, Train_batch.shape[2]), device=device)
      if training_method == 'recursive':
        for i in range(n_future):
          decoder_output, d_hn, d_cn = model.decoder(decoder_input, decoder_hidden, decoder_cell)
          outputs[:,i,:] = decoder_output
          decoder_input = decoder_output
          decoder_hidden = d_hn
          decoder_cell   = d_cn
      elif training_method == 'mtf':
        for i in range(n_future):
          decoder_output, d_hn, d_cn = model.decoder(decoder_input, decoder_hidden, decoder_cell)
          outputs[:,i,:] = decoder_output

          if random.random() < ratio:
              decoder_input = Truth_batch[:, i, :]
          else:
              decoder_input  = decoder_output

          decoder_hidden = d_hn
          decoder_cell   = d_cn

      # compute the loss
      loss = criterion(outputs, Truth_batch)
      epo_train_loss += loss.item()

      # compute the accuracy by r2 stats
      acc_r2 = 1.0 - loss/criterion(Truth_batch, torch.mean(Truth_batch) + torch.zeros_like(Truth_batch, device=device))
      epo_train_acc_r2 += acc_r2.item()

      # compute the accuracy by rel error
      acc_rel = 1.0 - loss/criterion(Truth_batch, torch.zeros_like(Truth_batch, device=device))
      epo_train_acc_rel += acc_rel.item()
      
      # back-prop
      loss.backward()
      optimizer.step()

      # update teacher's forcing ratio
      inc    = 0.005
      if ratio > inc:
        ratio = ratio-inc

  return epo_train_loss, epo_train_acc_r2, epo_train_acc_rel, model

# test the model
def model_test(device, model, testloader, criterion, n_future):
  
  epo_test_loss = 0
  epo_test_acc_rel  = 0
  epo_test_acc_r2   = 0

  model.eval()

  with torch.no_grad():
    for Test_batch, Test_Truth_batch in testloader:
      e_hn,e_cn =  model.encoder(Test_batch)
      outputs = torch.zeros((Test_batch.shape[0], n_future, Test_batch.shape[2]), device=device)
      
      # decode input_tensor to have output
      decoder_input = Test_batch[:, -1, :]
      decoder_hidden = e_hn
      decoder_cell   = e_cn

      for i in range(n_future):
        decoder_output, d_hn, d_cn = model.decoder(decoder_input, decoder_hidden, decoder_cell)
        outputs[:,i,:]  = decoder_output
        decoder_input = decoder_output
        decoder_hidden = d_hn
        decoder_cell   = d_cn

      # compute the loss
      loss = criterion(outputs,Test_Truth_batch)
      epo_test_loss = epo_test_loss + loss.item()

      # compute the accuracy by r2 stats
      acc_r2 = 1.0 - loss/criterion(Test_Truth_batch, torch.mean(Test_Truth_batch) + torch.zeros_like(Test_Truth_batch, device=device))
      epo_test_acc_r2+= acc_r2.item()

      # compute the accuracy by relative error
      acc_rel = 1.0 - loss/criterion(Test_Truth_batch,  torch.zeros_like(Test_Truth_batch, device=device))
      epo_test_acc_rel += acc_rel.item()
  
  return epo_test_loss, epo_test_acc_r2, epo_test_acc_rel

# Do prediction
def model_predict(device, model, X, n_future):
  
  model.eval()
  X = X.unsqueeze(0) # add single batch dimension

  # encode input_tensor
  e_hn,e_cn =  model.encoder(X)
  outputs = torch.zeros((n_future, X.shape[2]), device=device)

  # decode input_tensor to have output
  decoder_input = X[:, -1, :]
  decoder_hidden = e_hn
  decoder_cell   = e_cn

  for i in range(n_future):
      decoder_output, d_hn, d_cn = model.decoder(decoder_input, decoder_hidden, decoder_cell)
      outputs[i,:] = decoder_output
      decoder_input = decoder_output
      decoder_hidden = d_hn
      decoder_cell   = d_cn

  return outputs



# Functions to be used
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

# scale all the values in the tensor to [0,1] and return additional max, min values for future inverse scaling
def Scale_to_zero_one(X,Y):
    X_min = X.min()
    X_max = X.max()
    Y_min = Y.min()
    Y_max = Y.max()

    scale_min, scale_max = min(X_min, Y_min), max(X_max, Y_max)

    X = (X - scale_max) / (-scale_min+scale_max)
    Y = (Y - scale_max) / (-scale_min+scale_max)
    return X,Y,scale_max, scale_min

# scale them back to original scales, use the global max and min
def scale_forward(X, scale_max, scale_min):
  X = (X - scale_max) / (-scale_min+scale_max)
  return X

# scale them back to original scales
def scale_it_back(X,scale_max,scale_min):
  X = X*(scale_max-scale_min) + scale_max
  return X


# Use the real displacement data for all the nodes
def Dis_data_filtered_full(device, input_size, filter_size, n_past, n_future, Path, cut_off):
  
  with h5py.File(Path,'r') as f:
    Data_numpy = f['Displacement']
    Data_numpy = Data_numpy[0:int(cut_off*len(Data_numpy)),:]

  Data_numpy = Data_numpy[0::filter_size,:]

  # convert it to tensor
  Data_numpy = torch.from_numpy(Data_numpy).float().to(device)

  # number of samples, one sample is n_past in the past and n_future in the future
  total_groups = Data_numpy.shape[0]- n_future - n_past + 1

  # pre-allocate the memory directly on gpu (batch first)
  trainX = torch.zeros((total_groups, n_past, input_size), device=device)
  trainY = torch.zeros((total_groups, n_future, input_size), device=device)

  # windowing
  counter = 0
  for i in range(n_past, Data_numpy.shape[0]- n_future + 1):
      trainX[counter,:,:] = Data_numpy[i- n_past:i,:]
      trainY[counter,:,:] = Data_numpy[i:i+n_future,:]
      counter = counter + 1
    
  TrainX_total = trainX
  TrainY_total = trainY

  return TrainX_total, TrainY_total


# Use the real displacement data for a subset, named coronary but for a general usage
def Dis_data_filtered_subset_coronary(device, input_size, filter_size, n_past, n_future, Path, cut_off):
  
  with h5py.File(Path,'r') as f:
    Data_numpy = np.array(f['Displacement'])
  
  Data_numpy = Data_numpy.transpose()
  Data_numpy = Data_numpy[0:int(cut_off*len(Data_numpy)),:] # only the first half
  Data_numpy = Data_numpy[0::filter_size,:] # apply filters

  # convert it to tensor
  Data_numpy = torch.from_numpy(Data_numpy).float().to(device)

  # number of samples, one sample is n_past in the past and n_future in the future
  total_groups = Data_numpy.shape[0]- n_future - n_past + 1

  # pre-allocate the memory directly on gpu (batch first)
  trainX = torch.zeros((total_groups, n_past, input_size), device=device)
  trainY = torch.zeros((total_groups, n_future, input_size), device=device)

  # windowing
  counter = 0
  for i in range(n_past, Data_numpy.shape[0]- n_future + 1):
      trainX[counter,:,:] = Data_numpy[i- n_past:i,:]
      trainY[counter,:,:] = Data_numpy[i:i+n_future,:]
      counter = counter + 1
    
  TrainX_total = trainX
  TrainY_total = trainY

  return TrainX_total, TrainY_total
