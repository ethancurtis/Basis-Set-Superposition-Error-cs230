import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset

import numpy as np
import os
import helper_funcs as hf

class InMemMolecule(InMemoryDataset):

	def __init__(self, root=None, partition="debug",newData = False, transform=None, pre_transform=None):
		assert(partition in ['train','dev','test','debug'])
		path = ''
		if partition == 'train':
			path = r'/home/curtie/4_bsse/0_bbcp_database/1_bbcp_database'
		elif partition == 'dev':
			path = r'/home/curtie/4_bsse/0_bbcp_database/5_dev_set'
		elif partition == 'test':
			path = r'/home/curtie/4_bsse/0_bbcp_database/6_test_set'
		elif partition == 'debug':
			path=r'/home/curtie/4_bsse/1_gcp/3_calc_gcp_on_database'
		self.root = path
		self.xyzList = []
		if newData:
			self.xyzList = hf.getXYZList(os.path.join(path,self.raw_dir)) 
			print("Dataset " + partition + " has " + str(len(self.xyzList)) + " elements")
		super().__init__(path)
		self.data, self.slices = torch.load(self.processed_paths[0])
		

	@property
	def processed_file_names(self):
		return ['data.torch'] + [ mol.replace("xyz","torch") for mol in self.xyzList ]

	@property 
	def raw_file_names(self):
		return self.xyzList

	def process(self):
		dataset = []
		for mol in self.xyzList:
			savePath = os.path.join(self.root, self.processed_dir, mol.replace("xyz","torch"))
			if not os.path.isfile(savePath):
				molData = hf.makeData(os.path.join(self.root, self.raw_dir, mol))
				torch.save(molData,savePath)
				dataset.append(molData)
			else:
				molData = torch.load(savePath)
				dataset.append(molData)

		data, slices = self.collate(dataset)
		torch.save((data, slices), self.processed_paths[0])

class NormalizedMolecule(InMemoryDataset):

	def __init__(self, root=None, partition="debug",newData = False, transform=None, pre_transform=None):
		assert(partition in ['train','dev','test','debug'])
		path = ''
		if partition == 'train':
			path = r'/home/curtie/4_bsse/0_bbcp_database/1_bbcp_database'
		elif partition == 'dev':
			path = r'/home/curtie/4_bsse/0_bbcp_database/5_dev_set'
		elif partition == 'test':
			path = r'/home/curtie/4_bsse/0_bbcp_database/6_test_set'
		elif partition == 'debug':
			path=r'/home/curtie/4_bsse/1_gcp/3_calc_gcp_on_database'
		self.root = path
		self.xyzList = []
		if newData:
			self.xyzList = hf.getXYZList(os.path.join(path,self.raw_dir)) 
			print("Dataset " + partition + " has " + str(len(self.xyzList)) + " elements")
		super().__init__(path)
		self.data, self.slices = torch.load(self.processed_paths[0])
		

	@property
	def processed_file_names(self):
		return ['normalized.torch'] + [ mol.replace("xyz","torch") for mol in self.xyzList ]

	@property 
	def raw_file_names(self):
		return self.xyzList

	def process(self):
		dataset = []
		labels = []
		for mol in self.xyzList:
			savePath = os.path.join(self.root, self.processed_dir, mol.replace("xyz","torch"))
			if not os.path.isfile(savePath):
				molData = hf.makeData(os.path.join(self.root, self.raw_dir, mol))
				torch.save(molData,savePath)
				labels.append(molData.y)
			else:
				molData = torch.load(savePath)
				labels.append(molData.y)
		labels = np.asarray(labels,dtype=np.float64)
		mean = np.mean(labels)
		stdev = np.std(labels)
		print("dataset mean: ",mean)
		print("dataset stdev: ",stdev)
		for mol in self.xyzList:
			savePath = os.path.join(self.root, self.processed_dir, mol.replace("xyz","torch"))
			molData = torch.load(savePath)
			molData.y = torch.tensor(np.asarray([(molData.y - mean) / stdev],dtype=np.float64), dtype=torch.double)
			dataset.append(molData)

		data, slices = self.collate(dataset)
		torch.save((data, slices), self.processed_paths[0])
