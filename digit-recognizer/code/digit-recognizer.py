from sklearn import datasets
import pylab as pl 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import metrics
import re
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn import svm

from sklearn.externals import joblib
from collections import Counter


delimiter = ","
def loadData(fileName):
	f = open(fileName);
	count = 0;
	features =[]
	target =[]
	for line in f:
		
		line = line.strip()
		fields = re.split(delimiter,line)
		if count == 0:
			count +=1; # Skip the first line
			continue;
		targetClass = fields[0]	
		featureVector = fields[1:];
		features.append([int(i) for i in featureVector]);
		target.append(int(targetClass));
		
	return (features,target)

def train():
	"""docstring for main"""
	filename = "../data/train.csv"
	print "Loading Data"
	(train_X, train_y) = loadData(filename)
	print "Data loaded"
	#clf = svm.SVC(verbose=True);
	clf = ExtraTreesClassifier(n_estimators=10000, max_depth=None,min_samples_split=1, random_state=0,compute_importances=True)
	print "Training..."
	clf.fit(train_X, train_y) 
	print "Training complete"
	#clf = KNeighborsClassifier(n_neighbors=10);
	# clf = RandomForestClassifier(n_estimators=117)
	#clf = linear_model.LogisticRegression(C=1e5, penalty="l1");
	print "Cross validation started..."
	scores = cross_validation.cross_val_score(clf, train_X, train_y, cv=3) # rbf = 74, 3degpoly = 
	print "Accuracy = %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2);
	print "Cross validation complete..."
	joblib.dump(clf, './ExtraTrees.pkl')
	return clf;

def test(clf, model=""):
	"""docstring for test"""
	if (model == ""):
		clf = train();
	else:
		#filename = "../data/train.csv"
		#(train_X, train_y) = loadData(filename)
		
		clf = joblib.load(model)
		#scores = cross_validation.cross_val_score(clf, train_X, train_y, cv=3) # rbf = 74, 3degpoly = 
		#print "Accuracy = %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2);
		
	print dir(clf)
	clf.n_jobs = 1;
	filename = "../data/test.csv"
	fin = open(filename)
	fout = open("./resultom.txt","w");
	count=0;
	featureVector =[]
	testSamples = []
	for line in fin:
		
		line = line.strip()
		fields = re.split(delimiter,line)
		if count == 0:
			count +=1; # Skip the first line
			continue;
		featureVector = [int(i) for i in fields];
		#print featureVector	
		op = clf.predict(featureVector);
		data = Counter(op)
		#print data.most_common(1)[0][0];
		fout.write(str(int(data.most_common(1)[0][0]))+"\n")
	fout.close()
	pass	
	

def main():
	"""docstring for main"""
	#clf = train();
	test(clf=None,model="");
	print "Done"
	pass	
	pass
	

if __name__ == '__main__':
	main()