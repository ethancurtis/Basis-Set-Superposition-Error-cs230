import numpy as np
import torch
from torch_geometric.data import Data
import os
import fnmatch


def readXYZ(filepath):
	xyz = []
	atoms = []
	with open(filepath, 'r') as f:
		n = int(f.readline())
		label = float(f.readline()) * 627.509 
		for line in f.readlines():
			line = line.split()
			xyz.append(line[1:])
			if line[0] == 'H':
				atoms.append(1)
			elif line[0] == 'C':
				atoms.append(6)
			elif line[0] == 'N':
				atoms.append(7)
			elif line[0] == 'O':
				atoms.append(8)
			elif line[0] == 'F':
				atoms.append(9)
			elif line[0] == 'S':
				atoms.append(16)
			elif line[0] == 'Cl':
				atoms.append(17)
			else:
				raise Exception("UNRECOGNIZED ATOM TYPE")
	
	xyz = np.asarray(xyz, dtype=np.float32)
	atoms = np.asarray(atoms, dtype=np.int16)
	inMol1Heteroatoms = True
	i = 0
	while i < n:
		if atoms[i] == 1:
			inMol1Heteroatoms = False
		if not inMol1Heteroatoms and atoms[i] != 1:
			break
		i+=1
	n1 = i
	n2 = n - n1
	return xyz, atoms, label, n1, n2

def buildEdgeList(xyz, n1, n2, minCut = 0, maxCut = 9):
	dist = []
	edgeIndex = []
	for i in np.arange(n1):
		for j in np.arange(n2):
			r = np.linalg.norm(xyz[i,:] - xyz[n1+j,:])
			if r > minCut and r < maxCut:
				dist.append(r)
				edgeIndex.append([i,j+n1])
	dist = np.asarray(dist, dtype=np.float32)
	edgeIndex = np.asarray(edgeIndex, dtype=np.int16)
	return dist, edgeIndex
		
def oneHotLabelAtoms(atoms, atomList = [1,6,7,8,9,16,17]):
	atomLen = np.shape(atoms)[0]
	listLen = len(atomList)
	oneHotAtoms = np.zeros((atomLen,listLen))
	for i in np.arange(atomLen):
		for j in np.arange(listLen):
			if atoms[i] == atomList[j]:
				oneHotAtoms[i,j] = 1.0
				continue
	return oneHotAtoms

def makeData(filepath):
	xyz, atoms, label, n1, n2 = readXYZ(filepath)
	dist, edgeIndex = buildEdgeList(xyz, n1, n2)
	oneHotAtoms = oneHotLabelAtoms(atoms)
	data = Data()
	data.num_nodes = np.shape(atoms)[0]
	data.edge_index = torch.tensor(edgeIndex.T, dtype=torch.long)
	data.edge_attr = torch.tensor(dist, dtype=torch.float).view(-1,1)
	data.y = label
	data.node_attr = torch.tensor(oneHotAtoms, dtype=torch.float)
	return data

def getXYZList(path):
	xyzList = []
	for entry in os.listdir(path):
		if fnmatch.fnmatch(entry,"*xyz"):
			with open(os.path.join(path,entry),'r') as f:
				line = f.readline()
				try:
					int(line)
				except:
					continue
				line = f.readline()
				try:
					bbcp = float(line)
					if bbcp < 1.0 and bbcp > 0:
						xyzList.append(entry)
				except:
					continue
	return xyzList
