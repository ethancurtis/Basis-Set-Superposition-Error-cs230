#!/home/curtie/.conda/envs/torch_env/bin/python

import datasets as d
import models
import torch_geometric as pyg
import torch_funcs
import torch
import torch.nn as nn
import callbacks
from callbacks import Checkpoints, EarlyStopping

trainSet = d.InMemMolecule(partition='train')
trainLoader = pyg.data.DataLoader(trainSet, shuffle = True, batch_size = 64)
valSet = d.InMemMolecule(partition='dev')
valLoader = pyg.data.DataLoader(valSet, shuffle = True, batch_size = 64)

model = models.Model1(input_dimension=15)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss = nn.MSELoss()
earlyStopping = EarlyStopping(patience=100)
checkpoints = Checkpoints(checkpoint_start=0)
callback_list = [earlyStopping, checkpoints]
history = torch_funcs.train_model(model, loss, optimizer, trainLoader, valLoader, n_epochs=20, callbacks = callback_list, output = 'long')
torch_funcs.plot_history(history, size=(7,5))


