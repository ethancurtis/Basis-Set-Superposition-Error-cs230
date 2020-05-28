import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from torch_scatter import segment_coo


class PygModel(nn.Module):
	def __init__(self):
		super().__init__()

	def unpack_batch(self, batch):
		return batch, batch.y

class Model1(PygModel):
	def __init__(self, input_dimension):
		super(Model1, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 10)
		self.lin1 = nn.Linear(10, 5)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(5, 5)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act2(self.lin2(x))
		x = self.lin3(x)
		return x
