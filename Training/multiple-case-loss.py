import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 20})

num_epochs = 3001
num_cases  = 3
dpi = 100

# plot the losses in a loglog fashion
fig1 = plt.figure(figsize=(8, 8), dpi=dpi)
for i in range(num_cases):
	f1  = 'Case'+str(i+1)+'-Trained_model/Train_loss.csv'
	train_loss_save      = genfromtxt(f1, delimiter = ',')
	plt.loglog(range(num_epochs), train_loss_save,label='Cases='+str(i+1))

plt.title('Training')
plt.legend()
plt.ylim(1e-11, 1e-1)



fig2 = plt.figure(figsize=(8, 8), dpi=dpi)
for i in range(num_cases):
	f2  = 'Case'+str(i+1)+'-Trained_model/Test_loss.csv'
	test_loss_save      = genfromtxt(f2, delimiter = ',')
	plt.loglog(range(num_epochs), test_loss_save,label='Cases='+str(i+1))
plt.legend()
plt.title('Testing')
plt.ylim(1e-11, 1e-1)
plt.show()
