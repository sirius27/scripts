import torch as t
from torch import nn
fc_input_size = 20
fc_hidden_sizes = [15, 12, 10]
num_classes = 5

kernel_num = 5
kernel_size = 5
kernel_sizes = [5,4,3]
class MyNN(nn.Module):
	def __init__(self):
		super(MyNN, self).__init__()
		'''
		fcs = [nn.Sequential(
				 nn.Linear(fc_input_size, fc_hidden_size),
				 nn.ReLU(),
				 nn.Linear(fc_hidden_size,num_classes)
		) for fc_hidden_size in fc_hidden_sizes]
		#self.network = nn.ModuleList(fcs)
		self.network = fcs
		'''
		self.conv1 = nn.Conv1d(in_channels = 3, out_channels = kernel_num, kernel_size = kernel_size)
		self.max_pool1
	def forward(self, X):
		#z1 = self.fc1(X)
		#h1 = self.relu(z1)
		#outputs = [network(X) for network in self.network]
		conv_outputs = self.conv1(X)
		print conv_outputs
		return conv_outputs

if __name__ == '__main__':
	m = MyNN()
	data = t.autograd.Variable(t.randn(5000,3,30))
	output = m(data)
