from torch import nn
import torch as t
import numpy as np

kernel_sizes = [1,2,3,4]
embedding_dim = 50
seq_length = 30
kernel_num = 100 #number of kernels in the same size
fc_hidden_size=10
num_classes=10
class MultiCNN(nn.Module):
	def __init__(self):
		super(MultiCNN, self).__init__()
		convs = [ nn.Sequential(
						nn.Conv1d(in_channels = embedding_dim,
								  out_channels = kernel_num,
								  kernel_size = kernel_size),
						nn.BatchNorm1d(kernel_num),
						nn.ReLU(inplace=True),
						
						nn.Conv1d(in_channels = kernel_num,
								  out_channels = kernel_num,
								  kernel_size = kernel_size),
						nn.BatchNorm1d(kernel_num),
						nn.ReLU(inplace=True),
						nn.MaxPool1d(kernel_size = seq_length - kernel_size * 2 + 2)
						)
				for kernel_size in kernel_sizes]

		self.convs = nn.ModuleList(convs)	
		self.fc = nn.Sequential(
			nn.Linear(len(kernel_sizes) * kernel_num, fc_hidden_size),
			nn.BatchNorm1d(fc_hidden_size),
			nn.ReLU(inplace=True),
			nn.Linear(fc_hidden_size, num_classes)
		)

	def forward(self, articles):
		outputs = [conv(articles) for conv in self.convs]
		conv_out = t.cat(outputs, dim=1)
		reshaped = conv_out.view(conv_out.size(0),-1)
		logits = self.fc(reshaped)
		return logits
	
	def fit(self, X,y):
		self.
if __name__ == '__main__':
	m = MultiCNN()
	#X = t.autograd.Variable(t.arange(0,5000).view(100,50)).long()
	X = t.autograd.Variable(t.randn(100,50,30))
	o = m(X)
	print o.size()
