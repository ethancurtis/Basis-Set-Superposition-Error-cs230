import numpy as np
import subprocess
import os
import fnmatch
import random
import math

def randomMiniBatches(filenames, labels, mbSize = 64):
	m = len(filenames)

	minibatches = []
	mbLabels = []
	labelsList = labels.tolist()
	state = random.getstate()
	random.shuffle(filenames)
	random.setstate(state)
	random.shuffle(labelsList)
	numCompleteBatches = int(m/mbSize)
	labels = np.asarray(labelsList,dtype=np.float32)
	for i in range(numCompleteBatches):
		minibatch = filenames[mbSize * i : mbSize * (i + 1)]
		minibatches.append(minibatch)
		mbLabel = labels[mbSize * i : mbSize * (i + 1)]
		mbLabels.append(mbLabel)

	if m % mbSize != 0:
		minibatch = filenames[mbSize * numCompleteBatches :]
		minibatches.append(minibatch)
		mbLabel = labels[mbSize * numCompleteBatches :]
		mbLabels.append(mbLabel)
	return minibatches, mbLabels

def getGcpGrads(path, paramFile, minibatch, doEmiss = False, doZa = False):
	command = ['./gcpParamGrad.exe', paramFile]
	minibatch = [ path + name for name in minibatch ]
	gcpOutput = subprocess.run(command+minibatch,stdout=subprocess.PIPE,text=True).stdout.split("\n")
	dimerGrad = []
	for i in np.arange(len(minibatch) * (1 + doEmiss + doZa)):
		temp = gcpOutput[i].split()
		if doEmiss:
			temp += gcpOutput[i+1].split()
		if doZa:
			temp += gcpOutput[i+1+doEmiss].split()
		i += doEmiss + doZa
		dimerGrad.append(temp)
	minibatchA = [ name + "_A" for name in minibatch ]
	gcpOutput = subprocess.run(command+minibatchA,stdout=subprocess.PIPE,text=True).stdout.split("\n")
	aGrad = []
	for i in np.arange(len(minibatch) * (1 + doEmiss + doZa)):
		temp = gcpOutput[i].split()
		if doEmiss:
			temp += gcpOutput[i+1].split()
		if doZa:
			temp += gcpOutput[i+1+doEmiss].split()
		i += doEmiss + doZa
		aGrad.append(temp)
	minibatchB = [ name + "_B" for name in minibatch ]
	gcpOutput = subprocess.run(command+minibatchB,stdout=subprocess.PIPE,text=True).stdout.split("\n")
	bGrad = []
	for i in np.arange(len(minibatch) * (1 + doEmiss + doZa)):
		temp = gcpOutput[i].split()
		if doEmiss:
			temp += gcpOutput[i+1].split()
		if doZa:
			temp += gcpOutput[i+1+doEmiss].split()
		i += doEmiss + doZa
		bGrad.append(temp)
	
	return np.asarray(dimerGrad,dtype=np.float32) - np.asarray(aGrad, dtype=np.float32) - np.asarray(bGrad, dtype=np.float32)

def getGcpEnergies(path, paramFile, minibatch):
	command = ['./gcpEnergy.exe', paramFile]
	minibatch = [path+name for name in minibatch]
	dimerGcp = np.asarray(subprocess.run(command+minibatch,stdout=subprocess.PIPE,text=True).stdout.split(),dtype=np.float32)
	minibatchA = [ name+"_A" for name in minibatch ]
	aGcp = np.asarray(subprocess.run(command+minibatchA,stdout=subprocess.PIPE,text=True).stdout.split(), dtype=np.float32)
	minibatchB = [ name+"_B" for name in minibatch ]
	bGcp = np.asarray(subprocess.run(command+minibatchB,stdout=subprocess.PIPE,text=True).stdout.split(), dtype=np.float32)
	return dimerGcp - aGcp - bGcp

def adamUpdate(params, grads, v, s, t, learnRate = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
	v = beta1 * v + (1 - beta1) * grads
	v = v / (1 - beta1**t)
	s = beta2 * s + (1 - beta2) * grads * grads
	s = s / (1 - beta2**t)
	params -= learnRate * v / np.sqrt(s + eps)
	return params, v, s

def computeParamGrads(grads, errors):
	return np.mean(errors[:,None]*grads,axis=0)

def computeErrors(gcp, bbcp):
	return (bbcp - gcp)

def getLabels(path,filenames):
	labels = []
	badxyzs = []
	for xyz in filenames:
		with open(path+xyz,'r') as f:
			f.readline()
			line = f.readline()
			try:
				label = -1 * float(line)
				if abs(label) > 1:
					badxyzs.append(xyz)
				else:
					labels.append(label)
			except:
				badxyzs.append(xyz)
	for xyz in badxyzs:
		filenames.remove(xyz)
	return np.asarray(labels, dtype=np.float32)

def writeParamFile(params, paramFile, doEmiss = False, doZa = False):
	with open(paramFile,'w') as f:
		f.write("%f %f %f %f" %(params[0], params[1], params[2], params[3]))
		if doEmiss:
			f.write("%f %f %f %f %f %f %f" %(params[4], params[5], params[6], params[7], params[8], params[9], params[10]))
		if doZa:
			f.write("%f %f %f %f %f %f %f" %(params[4+7*doEmiss], params[5+7*doEmiss], params[6+7*doEmiss], params[8+7*doEmiss], params[9+7*doEmiss], params[10+7*doEmiss]))

def createABXYZs(filenames, path):
	badxyzs = []
	for xyz in filenames:
		if not os.path.isfile(path+xyz+"_A"):
			with open(path + xyz,'r') as f:
				try:
					n = int(f.readline())
				except:
					badxyzs.append(xyz)
					continue
				f.readline()
				inMol1Heteroatoms = True
				inMol1 = True
				xyz1 = []
				xyz2 = []
				for line in f.readlines():
					test = line
					if test.split()[0] == "H":
						inMol1Heteroatoms = False
					if test.split()[0] != "H" and not inMol1Heteroatoms:
						inMol1 = False
					if inMol1:
						xyz1.append(line)
					if not inMol1:
						xyz2.append(line)
			with open(path + xyz + "_A",'w') as f:
				f.write(str(len(xyz1))+"\n\n")
				for line in xyz1:
					f.write(str(line))
			with open(path + xyz + "_B",'w') as f:
				f.write(str(len(xyz2))+"\n\n")
				for line in xyz2:
					f.write(str(line))
	for xyz in badxyzs:
		filenames.remove(xyz)
						
def getXYZList(path):
	xyzList = []
	for entry in os.listdir(path):
		if fnmatch.fnmatch(entry,"*xyz"):
			xyzList.append(entry)
			with open(os.path.join(path,entry),'r') as f:
				line = f.readline()
				try:
					int(line)
				except:
					continue
				line = f.readline()
				try:
					bbcp = float(line)
					if bbcp < 0.1 and bbcp > -1.0:
						xyzList.append(entry)
				except:
					continue
	return xyzList	

def readParamFile(paramFilePath, doEmiss = False, doZa = False):
	with open(paramFilePath,'r') as f:
		params = f.readline().split()
		if doEmiss:
			params += f.readline().split()
		if doZa:
			params += f.readline().split()
	return np.asarray(params,dtype=np.float32)

def evaluateEnergy(paramFilePath, path, doEmiss = False, doZa = False):
	params = readParamFile(paramFilePath,doEmiss,doZa)
	xyzList = getXYZList(path)
	labels = getLabels(path, xyzList)
	createABXYZs(xyzList, path)
	gcp = getGcpEnergies(path, paramFilePath, xyzList)
	meanY = np.mean(labels)
	stdevY = np.std(labels)
	errors = computeErrors(gcp, labels)
	mae = np.mean(np.abs(errors))
	me = np.mean(errors)
	stdevE = np.std(errors)
	mre = np.mean(np.abs(errors) / labels)
	mse = np.mean(errors**2)
	print("Mean BBCP energy ", meanY)
	print("BBCP stdev ", stdevY)
	print("MAE ", mae)
	print("ME ", me)
	print("Error stdev", stdevE)
	print("MRE ", mre)
	print("MSE ", mse)
	

def adamOptimize(path, params, learnRate = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, batchSize = 64, numEpochs = 100, doEmiss = False, doZa = False, paramFile = 'params.txt', convthre = 1e-5):
	filenames = getXYZList(path)
	v = np.zeros(params.shape)
	s = np.zeros(params.shape)
	createABXYZs(filenames,path)
	labels = getLabels(path, filenames)
	t = 0
	previousCost = 0

	writeParamFile(params, paramFile, doEmiss, doZa)

	for i in range(numEpochs):
		minibatches, mbLabels = randomMiniBatches(filenames, labels, batchSize)
		costTotal = 0
		
		for minibatch, mbLabel in zip(minibatches, mbLabels):
			gcp = getGcpEnergies(path, paramFile, minibatch)
			#print("gcp e",gcp)
			gcpGrad = getGcpGrads(path, paramFile, minibatch, doEmiss, doZa)
			errors = computeErrors(mbLabel, gcp)	
			#print("errors ",errors)
			paramGrad = computeParamGrads(gcpGrad,errors)	
			t += 1
			params, v, s = adamUpdate(params, paramGrad, v, s, t, learnRate, beta1, beta2, eps)
			costTotal += 0.5 * np.sum(errors**2)
			#print("minibatch cost", costTotal)
			writeParamFile(params, paramFile, doEmiss, doZa)
	
		print("Cost after epoch %d is %f" %(i, costTotal / len(filenames)))
		if abs(costTotal-previousCost) < convthre:
			break
		previousCost = costTotal
		writeParamFile(params, "epoch_"+str(i)+paramFile, doEmiss, doZa)
	
	writeParamFile(params, paramFile, doEmiss, doZa)


#path = '/home/curtie/4_bsse/0_bbcp_database/1_bbcp_database/raw/'
#path = '/home/curtie/4_bsse/1_gcp/3_calc_gcp_on_database/'
#params = np.array([0.1290, 1.1526, 1.1549, 1.1763])
#adamOptimize(path, params, numEpochs = 10, beta2 = 0.9)


path = '/home/curtie/4_bsse/0_bbcp_database/5_dev_set/raw/'
paramFile = '/home/curtie/4_bsse/1_gcp/params.txt'
evaluateEnergy(paramFile, path)
paramFile = '/home/curtie/4_bsse/1_gcp/original_params.txt'
evaluateEnergy(paramFile, path)
