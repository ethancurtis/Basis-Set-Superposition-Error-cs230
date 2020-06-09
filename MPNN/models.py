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

class Model2(PygModel):
	def __init__(self, input_dimension):
		super(Model2, self).__init__()
		self.mpl1 = layers.MPL_2(input_dimension = input_dimension, hidden_dimension = 10, output_dimension = 10)
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

class Model3(PygModel):
	def __init__(self, input_dimension):
		super(Model3, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 10)
		self.lin1 = nn.Linear(10, 10)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(10, 5)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(5, 5)
		self.act3 = nn.ReLU(10)
		self.lin4 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act3(self.lin3(x))
		x = self.lin4(x)
		return x

class Model4(PygModel):
	def __init__(self, input_dimension):
		super(Model4, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 20)
		self.lin1 = nn.Linear(20, 10)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(10, 10)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(10, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act2(self.lin2(x))
		x = self.lin3(x)
		return x

class Model5(PygModel):
	def __init__(self, input_dimension):
		super(Model5, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 10)
		self.lin1 = nn.Linear(10, 10)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(10, 5)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(5, 5)
		self.act3 = nn.ReLU(10)
		self.lin4 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act2(self.lin2(x))
		x = self.act3(self.lin3(x))
		x = self.lin4(x)
		return x

class Model6(PygModel):
	def __init__(self, input_dimension):
		super(Model6, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 10)
		self.lin1 = nn.Linear(10, 5)
		self.act1 = nn.Sigmoid()
		self.lin2 = nn.Linear(5, 5)
		self.act2 = nn.Sigmoid()
		self.lin3 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act2(self.lin2(x))
		x = self.lin3(x)
		return x

class Model7(PygModel):
	def __init__(self, input_dimension):
		super(Model7, self).__init__()
		self.mpl1 = layers.MPL_2(input_dimension = input_dimension, hidden_dimension = 10, output_dimension = 10)
		self.lin1 = nn.Linear(10, 10)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(10, 5)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(5, 5)
		self.act3 = nn.ReLU(10)
		self.lin4 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act3(self.lin3(x))
		x = self.lin4(x)
		return x

class Model8(PygModel):
	def __init__(self, input_dimension):
		super(Model8, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 10)
		self.lin1 = nn.Linear(10, 10)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(10, 5)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(5, 5)
		self.act3 = nn.ReLU(10)
		self.lin4 = nn.Linear(5, 5)
		self.act4 = nn.ReLU(10)
		self.lin5 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act3(self.lin3(x))
		x = self.act4(self.lin4(x))
		x = self.lin5(x)
		return x

class Model9(PygModel):
	def __init__(self, input_dimension):
		super(Model9, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 20)
		self.lin1 = nn.Linear(20, 20)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(20, 10)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(10, 10)
		self.act3 = nn.ReLU(10)
		self.lin4 = nn.Linear(10, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act3(self.lin3(x))
		x = self.lin4(x)
		return x

class Model10(PygModel):
	def __init__(self, input_dimension):
		super(Model10, self).__init__()
		self.mpl1 = layers.MPL_2(input_dimension = input_dimension, hidden_dimension = 10, output_dimension = 10)
		self.lin1 = nn.Linear(10, 10)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(10, 5)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(5, 5)
		self.act3 = nn.ReLU(10)
		self.lin4 = nn.Linear(5, 5)
		self.act4 = nn.ReLU(10)
		self.lin5 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act3(self.lin3(x))
		x = self.act4(self.lin4(x))
		x = self.lin5(x)
		return x

class Model11(PygModel):
	def __init__(self, input_dimension):
		super(Model11, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 10)
		self.lin1 = nn.Linear(10, 10)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(10, 10)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(10, 5)
		self.act3 = nn.ReLU(10)
		self.lin4 = nn.Linear(5, 5)
		self.act4 = nn.ReLU(10)
		self.lin5 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = self.act3(self.lin3(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act4(self.lin4(x))
		x = self.lin5(x)
		return x

class Model12(PygModel):
	def __init__(self, input_dimension):
		super(Model12, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 30)
		self.lin1 = nn.Linear(30, 20)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(20, 10)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(10, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act2(self.lin2(x))
		x = self.lin3(x)
		return x

class Model13(PygModel):
	def __init__(self, input_dimension):
		super(Model13, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 10)
		self.lin1 = nn.Linear(10, 10)
		self.act1 = nn.Sigmoid()
		self.lin2 = nn.Linear(10, 5)
		self.act2 = nn.Sigmoid()
		self.lin3 = nn.Linear(5, 5)
		self.act3 = nn.Sigmoid()
		self.lin4 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act3(self.lin3(x))
		x = self.lin4(x)
		return x

class Model14(PygModel):
	def __init__(self, input_dimension):
		super(Model14, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 30)
		self.lin1 = nn.Linear(30, 20)
		self.act1 = nn.ReLU(10)
		self.lin2 = nn.Linear(20, 10)
		self.act2 = nn.ReLU(10)
		self.lin3 = nn.Linear(10, 5)
		self.act3 = nn.ReLU(10)
		self.lin4 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act3(self.lin3(x))
		x = self.lin4(x)
		return x

class Model15(PygModel):
	def __init__(self, input_dimension):
		super(Model15, self).__init__()
		self.mpl1 = layers.MPL_1(input_dimension = input_dimension, output_dimension = 10)
		self.lin1 = nn.Linear(10, 10)
		self.act1 = nn.Tanh()
		self.lin2 = nn.Linear(10, 5)
		self.act2 = nn.Tanh()
		self.lin3 = nn.Linear(5, 5)
		self.act3 = nn.Tanh()
		self.lin4 = nn.Linear(5, 1)

	def forward(self, data):
		x = self.mpl1(data.edge_index, data.node_attr, data.edge_attr)
		x = self.act1(self.lin1(x))
		x = self.act2(self.lin2(x))
		x = segment_coo(x, data.batch, reduce="sum")
		x = self.act3(self.lin3(x))
		x = self.lin4(x)
		return x
