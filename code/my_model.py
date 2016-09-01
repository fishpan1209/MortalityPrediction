import utils
from sklearn import linear_model
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import *
from sklearn import metrics
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
import etl
from sklearn import neural_network
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier



#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	test_path = '../data/test/'
	
	events = pd.read_csv(os.path.join(test_path) + 'events.csv', parse_dates=['timestamp'])
	feature_map = pd.read_csv(os.path.join(test_path) + 'event_feature_map.csv')

	# create a fake mortality
	dead = pd.read_csv(os.path.join('../data/train/')+'mortality_events.csv',parse_dates=['timestamp']).ix[0:1,:]
	dead.set_value(0,'patient_id',123456789)
	dead.set_value(1,'patient_id',123456789)
	

	# create features
	deliverable_path='../data/test/'
	indx_date=etl.calculate_index_date(events, dead, deliverable_path)
	#indx_date=indx_date.ix[0:(indx_date.shape[0]-2),:]
	
	filtered_events=etl.filter_events(events, indx_date, deliverable_path)
	
	#print filtered_events
	patient_features, mortality_fake=etl.create_features(events, dead, feature_map)


   
	
	mortality={}
	for key in patient_features.keys():
		mortality[key]=0

	#print patient_features
	
	op_file='../deliverables/features_svmlight.test'
	op_deliverable='../deliverables/test_features.txt'
	etl.save_svmlight(patient_features, mortality, op_file, op_deliverable)


	X_test,Y_test = utils.get_data_from_svmlight('../deliverables/features_svmlight.test')
	


	
	return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
	

	
	lr = linear_model.LogisticRegression(C=10,random_state =545510477)
	
	
	clf=BaggingClassifier(lr,n_estimators=10)
	
	pred_list=np.zeros((X_test.shape[0], 10))
	for i in range(10):
		clf.fit(X_train, Y_train)
		pred_list[:,i]=clf.predict(X_test)
		pred=pd.DataFrame(pred_list)
		pred['result']=pred.mean(axis=1)
		print pred
		for i in range(pred.shape[0]):
			
			if pred['result'][i]<=0.5:
				pred.set_value(i,'result',float('0'))
			else:
				pred.set_value(i,'result',float('1'))
	pred=pred['result'].values
	print pred

	return pred
    



def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	