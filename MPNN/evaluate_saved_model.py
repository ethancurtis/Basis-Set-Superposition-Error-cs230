#!/home/curtie/.conda/envs/torch_env/bin/python

import datasets as d
import models
import torch_geometric as pyg
import torch_funcs
import torch
import torch.nn as nn
import callbacks
from callbacks import Checkpoints, EarlyStopping
import models

evalSet = d.InMemMolecule(partition='test')
evalLoader = pyg.data.DataLoader(evalSet, shuffle = True, batch_size = 64)
path = 'Checkpoints/model_14/checkpoint_e176_3.3226e-02.pt'
model_dict = torch.load(path)
model_state_dict = model_dict['model_state_dict']
model = models.Model14(input_dimension = 15)
model.load_state_dict(model_state_dict)
model.eval()
cLoss = 0
n = 0
loss = nn.MSELoss()
for batch in evalLoader:
	x, target = model.unpack_batch(batch)
	prediction = model(x)
	temploss = loss(prediction.view(-1,1), target.view(-1,1))
	if temploss.item() < .5:
		nBatch = target.shape[0]
		cLoss += temploss.item() * nBatch
		n += nBatch
print(cLoss / n, n)
