import models
import utils
from sklearn import cross_validation,preprocessing
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean
import numpy as np

from sklearn import linear_model
from sklearn.metrics import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import utils

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds

    kf=cross_validation.KFold(X.shape[0],n_folds=k)
    lr = linear_model.LogisticRegression(random_state =545510477)
    accuracy=[]
    auc=[]
    for train_indices, test_indices in kf:
    	
    	lr.fit(X[train_indices], Y[train_indices])
    	pred=lr.predict(X[test_indices])
    	acc=accuracy_score(pred, Y[test_indices])
    	accuracy.append(acc)
    	
    	aucurve = roc_auc_score(pred, Y[test_indices])
    	auc.append(aucurve)
    
    print mean(accuracy),mean(auc)

    
    return np.mean(accuracy),np.mean(auc)


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	rs=cross_validation.ShuffleSplit(len(Y),n_iter=iterNo,test_size=test_percent)
	lr = linear_model.LogisticRegression(random_state=545510477)
	accuracy=[]
	auc=[]
	for train_indices, test_indices in rs:
		X_train, X_test, Y_train, Y_test = X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]
		
		lr.fit(X_train, Y_train)
    	pred=lr.predict(X_test)
    	acc=accuracy_score(pred, Y_test)
    	accuracy.append(acc)
    	
    	aucurve = roc_auc_score(pred, Y_test)
    	auc.append(aucurve)


	return np.mean(accuracy),np.mean(auc)


def main():
	X,Y = utils.get_data_from_svmlight("../newtest/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

