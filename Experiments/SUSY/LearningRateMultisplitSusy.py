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
dir = os.path.dirname(__file__)

###Experiment Parameters
Path = 'LRMultiSplitSusy'
#Momentum iterations
mi = 8
numsplits = 5

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/susy.csv.gz"
dmatrix_train_filename = "susy_train.dmatrix"
dmatrix_test_filename = "susy_test.dmatrix"
dmatrix_val_filename = "susy_val.dmatrix"
csv_filename = "SUSY.csv"
train_pool = 0.6
test_pool = 0.4
test_val_split = 0.5

plot = True


# return xgboost dmatrix
def load_susy(r):
	if not os.path.isfile(csv_filename):
		print("please download susy file...")

	df_susy = pandas.read_csv(csv_filename, dtype=np.float32, header=None)
	X_train, X_test, y_train, y_test = train_test_split(df_susy.ix[:, 1:29], df_susy[0], test_size=test_pool, train_size=train_pool, random_state=r)
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
param['eta'] = 0.1
param['objective'] = 'binary:logitraw'
param['eval_metric'] = 'error'
param['silent'] = 1
param['tree_method'] = 'gpu_hist'

#Set axis and arrays to collect data
x_axis=range(1, numround + 1)

for s in range(0,numsplits):
	#Load new split
	dtrain, dtest, dval = load_susy(s)
	# specify validations set to watch performance
	watchlist = [(dtrain, 'train'), (dtest, 'test'), (dval, 'eval')]
	param['optimizer'] = 'momentum_optimizer'	
	###Training loop
	for m in range(0,2):
		M = 0
		res={}
		for c in range(0,mi):
			param['momentum'] = M
			res['res%s' % c] = {}
			booster = xgb.train(param, dtrain, numround, watchlist, evals_result=res['res%s' % c], early_stopping_rounds=esr)
			M = M + 0.1
			booster = None #Clear booster to stop GPU thrust
		Output((Path + '/'+str(s)+'Split/' + param['optimizer']), 1)
		param['optimizer'] = 'nesterov_optimizer'

