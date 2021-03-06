import xgboost as xgb
import numpy as np
import sys
import pandas
from io import StringIO
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import time
import json
import math
dir = os.path.dirname(__file__)

###Experiment Parameters
Path = 'LRSignTestCoverType'
#Momentum iterations
mi = 6
numsplits = 5

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

train_pool = 0.6
test_pool = 0.4
test_val_split = 0.5

plot = True


# return xgboost dmatrix
def load_cov(r):
	cov = fetch_covtype()
	X = cov.data
	y = cov.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pool, train_size=train_pool, random_state=r)
	dtrain = xgb.DMatrix(X_train, y_train)
	##Split test data into validation and test
	X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_val_split, train_size=test_val_split, random_state=r)
	dval = xgb.DMatrix(X_val, y_val)
	dtest = xgb.DMatrix(X_test, y_test)
	return dtrain, dtest, dval

def Output(path, t):
	#Make directory
	filepath = os.path.join(dir, path)
	if not os.path.exists(filepath):
   		os.makedirs(filepath)
	###Output raw data
	for c in range(0,mi):
		f = open(filepath + '/RawData%s.txt' % c, 'w')
		x = json.dumps(res['res%s' % c])
		f.write(x)

numround = 10000
esr = 100
param = {}
param['eta'] = 0.05
param['objective'] = 'multi:softmax'
param['silent'] = 1
param['num_class'] = 8
param['tree_method'] = 'gpu_hist'

#Set axis and arrays to collect data
x_axis=range(1, numround + 1)

baseRange = [1,1.5,2,2.5,math.e,3]
alphaRange = [0.9,0.92,0.94,0.96,0.98,1]


for s in range(0,numsplits):
	#Load new split
	dtrain, dtest, dval = load_cov(s)
	# specify validations set to watch performance
	watchlist = [(dtrain, 'train'), (dtest, 'test'), (dval, 'eval')]
	###Training loop
	for updateType in ['add_sign_optimizer', 'power_sign_optimizer']:
		param['optimizer'] = updateType
		res={}
		for c in range(0,mi):
			res['res%s' % c] = {}
			param['alpha'] = alphaRange[c]
			param['base'] = baseRange[c]
			booster = xgb.train(param, dtrain, numround, watchlist, evals_result=res['res%s' % c], early_stopping_rounds=esr)
			booster = None #Clear booster to stop GPU thrust
		Output((Path + '/'+ str(s)+'Split/' + param['optimizer']), 1)

