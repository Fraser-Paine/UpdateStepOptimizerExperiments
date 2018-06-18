import numpy as np
import sys
from io import StringIO
import matplotlib.pyplot as plt
from math import log10, floor
import json
import math

r = 5
numTests = 8
#Set eval type depending on data set to be graphed
#Regression (MSD) = "rmse", multiclass classification (Covertype) = "merror", Binary classification (HIGGS, SUSY) = "error"
evalType = "merror"

dataset = "Covertype"
learningRate = "03"

def round_sig(x, sig=2):
	return round(x, sig-int(floor(log10(abs(x))))-1)

colours = ['#e6194b', '#3cb44b', '#0082c8', '#ffe119', '#f58231', '#911eb4']
learningRates = []
M = {}

momRange = [0.0,0.1,0.2,0.3,0.4,0.5,0.6]
baseRange = [1,1.5,2,2.5,round_sig(math.e, sig=4),3]
alphaRange = [0.9,0.92,0.94,0.96,0.98,1]

def getMedian(v, opt):
	Vals = []
	for s in range(0,r):
		x = json.loads(open(('{0}Split/'+opt+'/RawData{1}.txt').format(str(s),str(v)), 'r').read())
		Vals.append(min(x["eval"][evalType]))
	i = Vals.index(np.percentile(Vals,50,interpolation='nearest'))
	return i, Vals[i]

#Get the data corresponding to a given tuning value and split index
def getData(s,v, opt):
	x = json.loads(open(('{0}Split/'+opt+'/RawData{1}.txt').format(str(s),str(v)), 'r').read())
	index = x["eval"][evalType].index(min(x["eval"][evalType]))
	TestError = x["test"][evalType][index]
	Data = x["test"][evalType]
	return Data, index, TestError

#find the value in the tuning range which contains the best median split score
def getBestMedian(rng, opt, skip = 0):
	minMedian = 999
	minV = None
	minS = None
	for v in range(0 + skip,len(rng)):
		nS, nMed = getMedian(v, opt)
		if nMed < minMedian:
			minMedian = nMed
			minV = v
			minS = nS
	return minV, minS


#Graph Setup
fig, ax1 = plt.subplots()

fig.suptitle('Test ' + evalType + ' found by early stopping on Validation error, learning rate of 0.3')
ax1.set_xlabel('Number of Iterations')
ax1.set_ylabel(evalType)

#Base
v, s = getBestMedian([0], 'momentum_optimizer')
d, i, t = getData(s,v,'momentum_optimizer')
ax1.plot(range(0, len(d)), d, colours[0], marker='D', markevery = [i], label = "Baseline, Test Error = " + str(t) + " for Baseline XGBoost, stopped at : " + str(i))

#Momentum
v, s = getBestMedian(momRange, 'momentum_optimizer', skip = 1)
d, i, t = getData(s,v,'momentum_optimizer')
val = momRange[v]
ax1.plot(range(0, len(d)), d, colours[1], marker='D', markevery = [i], label = "Momentum, Test Error = " + str(t) + " with " + str(val) + " momentum coefficent, stopped at : " + str(i))

#Nesterov
v, s = getBestMedian(momRange, 'nesterov_optimizer', skip = 1)
d, i, t = getData(s,v,'nesterov_optimizer')
val = momRange[v]
ax1.plot(range(0, len(d)), d, colours[2], marker='D', markevery = [i], label = "Nesterov, Test Error = " + str(t) + " with " + str(val) + " momentum coefficent, stopped at : " + str(i))

#Power Sign
v, s = getBestMedian(baseRange, 'power_sign_optimizer')
d, i, t = getData(s,v,'power_sign_optimizer')
val = baseRange[v]
ax1.plot(range(0, len(d)), d, colours[3], marker='D', markevery = [i], label = "PowerSign, Test Error = " + str(t) + " with " + str(val) + " Base value, stopped at : " + str(i))

#Add Sign
v, s = getBestMedian(alphaRange, 'add_sign_optimizer')
d, i, t = getData(s,v,'add_sign_optimizer')
val = alphaRange[v]
ax1.plot(range(0, len(d)), d, colours[4], marker='D', markevery = [i], label = "AddSign, Test Error = " + str(t) + " with " + str(val) + " Alpha value, stopped at : "  + str(i))


plt.legend()
fig.subplots_adjust(left=0.086, bottom=0.09, right=0.986, top=0.950)
fig.set_size_inches(8, 5)
plt.savefig('TrainingCurve'+ learningRate + dataset +'.png')
plt.show()

#plt.savefig('retrievedCoverType.png')
