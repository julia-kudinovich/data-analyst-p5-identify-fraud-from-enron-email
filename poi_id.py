
# coding: utf-8

# # Table of Contents
# * [Project 5 Identify Fraud from Enron Email](#Project-5-Identify-Fraud-from-Enron-Email)
# 	* [Question 1](#Question-1)
# 		* [Project Overview](#Project-Overview)
# 		* [Data Exploration](#Data-Exploration)
# 		* [Outliers](#Outliers)
# 	* [Question 2:](#Question-2:)
# 		* [Feature Selection](#Feature-Selection)
# 			* [Dealing with missing values](#Dealing-with-missing-values)
# 			* [New feature creation](#New-feature-creation)
# 		* [Feature scaling](#Feature-scaling)
# 	* [Question 3](#Question-3)
# 		* [Trying a variety of classifiers](#Trying-a-variety-of-classifiers)
# 			* [Decision Trees](#Decision-Trees)
# 			* [Naive Bayes](#Naive-Bayes)
# 			* [SVM](#SVM)
# 	* [Question 4](#Question-4)
# 		* [Parameter Tuning](#Parameter-Tuning)
# 	* [Question 5](#Question-5)
# 	* [Question 6](#Question-6)
# 	* [Dump the classifier, dataset, and features_list](#Dump-the-classifier,-dataset,-and-features_list)
# 

# # Project 5 Identify Fraud from Enron Email

# ## Question 1

# 
# >Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?
# 

# ### Project Overview

# Enron once was a one of the largest corporation in the energy sector. Fraudulent activities of upper management led Enron to collapse in 2002. 
# 
# In this project, I will be applying machine learning techniques to Enron dataset to identify persons of interest (POI) who may have committed fraud that lead to the Enron collapse. Enron dataset consists of financial and email data that was made publicly available after the Enron scandal.
# 
# There are 146 data points in the dataset that represent 146 upper management persons in the company. 18 persons out of the 146 are already identified as POIs. My goal will be to build a machine learning algorithms (POI identifier) based on financial and email data that is publicly available in the result of the Enron scandal. 
# 
# Machine learning is very useful in this kind of task as it can learn from the dataset, discover patterns in the data and classify new data based on the patterns.
# 

# In[217]:

###Importing libraries

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot
from tester import test_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.feature_selection import SelectKBest


# In[218]:

###  Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# ### Data Exploration

# In[159]:

print  'Number of data points (persons) in the data set:', len(data_dict)

#Number of features for each person
num_features=0
for person in data_dict:
    num_features = len(data_dict[person])
    break
print 'Number of features for each person: ', num_features

#Number of POIs (persons of interest)
num_poi=0
for person in data_dict:
    if data_dict[person]["poi"]==1:
        num_poi+=1
        
print 'Number of POIs (persons of interest):', num_poi

print '\n'

### Getting the list of all the features in the dataset
n=0
features=[]
for key, value in data_dict.iteritems():
    if n==0:
        features =  value.keys()
    n+=1
    
###Count how many NaN each feature has and output feature name + count
def countNaN(feature_name):
    count=0
    for person in data_dict:
        if data_dict[person][feature_name] == 'NaN':
            count += 1
            
    if count>0:           
        print 'Number of NaNs for', feature_name, ':', count
    
for f in features:
    countNaN(f)


# There are a lot of missing values in the data. handling of those will be discussed later in the analysis.

# ### Outliers

# Let's see if there are any outliers in the financial data. First, I want to plot salary and bonus.

# In[219]:

features = ['salary', 'bonus']

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# From the graph above we see that there is a very extreme outlier. Now, I will investigate who the outlier is and remove him from the dataset

# In[220]:

for person in data_dict:
    if data_dict[person]['salary']>2500000 and data_dict[person]['salary']!='NaN':
        print person


# It appears that the data set has data point for total values. This data point should be removed.

# In[221]:

#Remove 'Total' value
data_dict.pop( "TOTAL", 0 ) 
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# Now, the plot looks much better. However, there are still some outliers. Let's investigate further.
# 

# In[222]:

for key, value in data_dict.iteritems():
    name = key
    salary = value["salary"]
    bonus = value["bonus"]
    poi = value['poi']
    if (salary!= 'NaN' and bonus!='NaN') and (int(salary)>= 800000 and int(bonus)>=3000000):
        print name, ' - POI? -', poi


# Even though LAY KENNETH L and SKILLING JEFFREY K are outliers and we can remove them from the training dataset we can not do so for the test set. This way, we would have a training dataset that would generalize better, and would still be able to validate on the entire dataset. 
# 
# Having total value in the dataset made me curious if there are any non person data points in the set. Below I will output all the names to see any odd entries.

# In[223]:

for key, value in data_dict.iteritems():
    name = key
    print name


# Looking through the names I noticed `THE TRAVEL AGENCY IN THE PARK` which does not seem to represent a person. Moreover, data for the The travel Agency is mostly NaNs. I am excluding `THE TRAVEL AGENCY IN THE PARK` from the datatset as well.

# In[224]:

data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0 ) 


# Next I am going to check programmatically if there is anybody else who has all or almost all feature values missing.

# In[166]:

for key, value in data_dict.iteritems():
    name = key
    countNaN=0
    for feature in value:
        if (value[feature]=='NaN'):
            countNaN+=1
    print 'Number of missing falues for', name, ': ', countNaN 


# There are 21 features for each person in the dataset. And from above we see that there is a person that has 20 missing values (LOCKHART EUGENE E). I will remove this person form the dataset.

# In[225]:

data_dict.pop( "LOCKHART EUGENE E", 0 ) 


# ## Question 2: 

# >What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

# ### Feature Selection

# ####  Dealing with missing values

# In[226]:

### Getting the list of all the features in the dataset
n=0
features=[]
for key, value in data_dict.iteritems():
    if n==0:
        features =  value.keys()
    n+=1

###Count how many NaN missing values are related to POI or non POI
def countNaN(feature_name):
    count=0
    count_POI=0
    count_non_POI = 0
    for person in data_dict:
        if data_dict[person][feature_name] == 'NaN':
            count += 1
            if data_dict[person]['poi']==1:
                count_POI+=1
            elif data_dict[person]['poi']==0:
                count_non_POI+=1
    if count>0:           
        print 'Number of NaNs for', feature_name, ':', count
        print  "Percent of POI having NaNs:", round(100* count_POI/float(18),2), "%"
        print   "Percent of non POI having NaNs:", round(100* count_non_POI/float(146-18),2), "%"
        print "\n"
    
for f in features:
    countNaN(f)


# From above we see that there are several features with many missing values: `deferral_payments, restricted_stock_deferred, loan_advances,  director_fees ,deferred_income`. These features have around 100 and above missing values. Mentioned features will not be used in the algorithm since they do not contribute to finding patterns.
# 
# Moreover, features where missing values percent is very different between POI and non POI should also not be considered in the feature selection because the algorithm may identify a difference in NaN count as a pattern to distinguish a POI which is a wrong way to go. I will select only features having 30% or less difference between POI and not POI NaNs percentages. However, I will make an exclusion to this adding both salary and bonus to features list. This is solely based on my intuition since I believe that both salary and bonus are important features in this case.
# 
# Also, `email_address` should not be used as a feature because email is unique for each person and cannot be used to make distinction between POI or not POI. 
# 
# Preliminary feature list is below:
# 

# In[227]:

features_list_prelim = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'from_this_person_to_poi',
                 'from_poi_to_this_person', 'to_messages',  'long_term_incentive', 'shared_receipt_with_poi',
                        'from_messages',  'restricted_stock', 'total_payments']


# #### New feature creation

# Instead of using `from_this_person_to_poi` and `from_poi_to_this_person` directly I want to create 2 new features:
# proportion of `from_this_person_to_poi` and `from_poi_to_this_person` in total emails. Absolute value of emails to/from POI does not make much sense by itself. If one person has sent 10 emails to POI but his total emails sent is 20 the proportion is 0.5. While another person has also sent 10 emails but the total number sent is 100 making the proration 0.1. The first person is communicating more often with POI even though their total count of emails to POI is the same.

# In[256]:

for person in data_dict:
    if data_dict[person]['from_this_person_to_poi']!='NaN':
        data_dict[person]['from_this_person_to_poi_proportion']         = int(data_dict[person]['from_this_person_to_poi'])/float(data_dict[person]['from_messages'])
    else:
        data_dict[person]['from_this_person_to_poi_proportion']='NaN'
        
        
    if data_dict[person]['from_poi_to_this_person']!='NaN':
        data_dict[person]['from_poi_to_this_person_proportion']         = int(data_dict[person]['from_poi_to_this_person'])/float(data_dict[person]['to_messages'])
    else:
        data_dict[person]['from_poi_to_this_person_proportion']='NaN'


# I will test new features effect on the classification algorithm. I am going to run Decision Tree algorithm with and without new features and compare the results.

# In[257]:

features_list_prelim_wo_new = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'from_this_person_to_poi',
                 'from_poi_to_this_person', 'to_messages',  'long_term_incentive', 'shared_receipt_with_poi',
                        'from_messages',  'restricted_stock', 'total_payments']

features_list_prelim_w_new = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'from_this_person_to_poi',
                 'from_poi_to_this_person', 'to_messages',  'long_term_incentive', 'shared_receipt_with_poi',
                        'from_messages',  'restricted_stock', 'total_payments', 'from_poi_to_this_person_proportion',
                             'from_this_person_to_poi_proportion']


#Using test_classifier from tester.py
print 'Test Classifier without new features implementation:'
test_classifier(clf, my_dataset, features_list_prelim_wo_new , folds = 1000)

print 'Test Classifier with new features implementation:'
test_classifier(clf, my_dataset, features_list_prelim_w_new , folds = 1000)


# As we can see adding new features to the feature list improved all metrics, therefore new features creation is justified. I will add these new features to my preliminary features_list in place of  `'from_poi_to_this_person', 'from_this_person_to_poi'` because new created features are the better representation of volume of communication between person and POI.

# In[258]:

features_list_prelim = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'from_this_person_to_poi_proportion',
                 'from_poi_to_this_person_proportion', 'to_messages',  'long_term_incentive', 'shared_receipt_with_poi',
                        'from_messages',  'restricted_stock', 'total_payments']

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_prelim, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Next, I am using Select K Best automated feature selection function on my preliminary list to further narrow down features used.

# In[259]:

#Using k='all' do display all features
k_best = SelectKBest(k='all')
k_best.fit(features, labels)
for impt_num, impt in enumerate(k_best.scores_):
    print features_list_prelim[1+impt_num], impt


# From the result above it is seen that `to_messages and from_messages` have the lowest scores. I will exclude these 2 features form my feature_list. Let's check the effect of removing these 2 features on the classifier:

# In[260]:

features_list_prelim = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'from_this_person_to_poi_proportion',
                 'from_poi_to_this_person_proportion', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 
                        'total_payments']

print 'Test Classifier with to_messages and from_messages removed:'
test_classifier(clf, my_dataset, features_list_prelim, folds = 1000)


# All the metrics went up, therefore removal of  `to_messages and from_messages` is justified.
# 
# Final features list is below:

# In[231]:

features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'from_this_person_to_poi_proportion',
                 'from_poi_to_this_person_proportion', 'long_term_incentive', 'shared_receipt_with_poi','restricted_stock', 
                        'total_payments']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# ### Feature scaling

# I am not preforming any feature scaling because the algorithm I will be using do not require feature scaling

# ## Question 3

# >What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

# ### Trying a variety of classifiers

# #### Decision Trees

# In[232]:

#Implementing Decision Tree Classifier
clf=tree.DecisionTreeClassifier()

#Using test_classifier from tester.py
test_classifier(clf, my_dataset, features_list, folds = 1000)


# From above we see that the classifier has a pretty high accuracy score. However,recall is below 0.3.

# #### Naive Bayes

# In[233]:

clf = GaussianNB()

#Using test_classifier from tester.py
test_classifier(clf, my_dataset, features_list, folds = 1000)


# Naive Bayes while having a higher accuracy score has lower recall and F1 scores than Decision Tree algorithm. Precision are almost identical in both cases.

# #### SVM

# In[234]:

clf = SVC()

#Using test_classifier from tester.py
test_classifier(clf, my_dataset, features_list, folds = 1000)


# Support Vector Machines algortihm does not have enough true positives to make a prediction.
# 
# I will pick Decision Trees as my final algorithm. I do not need feature scaling since Naive Bayes do not require scaling.

# ## Question 4

# >What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?

# Algorithm may perform differently using different parameters. Tuning parameters means trying difference combination of parameters that make your algorithm perform at its best. Having optimal values for parameters will enable algorism to perform a learning in the best way possible. 
# 
# If you don’t tune your algorithm well, meaning you can over tune or under tune it, it may perform well on the training data but fail at testing data.
# 
# I will tune Decision tree algorithm using GridSearchCV.

# ### Parameter Tuning

# In[235]:

param_grid = {
                'min_samples_split': np.arange(2, 10),
                'splitter':('best','random'),
                'criterion': ('gini','entropy')
}

clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
clf.fit(features, labels)
print(clf.best_params_)


# Now run Decision Tree algorithm with suggested parameters.
# 

# In[236]:

clf=tree.DecisionTreeClassifier(min_samples_split = 2, 
                                    criterion ='gini',
                               splitter = 'random')

#Using test_classifier from teter.py
test_classifier(clf, my_dataset, features_list, folds = 1000)


# After tuning parameters all score went up. Now I have both precision and recall above 0.3.

# ## Question 5

# >What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

# Validation is checking how well the algorithm performs on the data it has never seen before. The way to check is to split the data into training and testing sets and access the accuracy score of prediction on a testing set.
# One of the classic mistakes you can make is to overfit your model. This happens when you have a small training data set or a lot of parameters in the model.
# 
#  One of the validation matrices is accuracy score. Let's split the dataset in to training and testing data and see the metrics.

# In[242]:

##Splitting the data into train and test
features_train, features_test, labels_train, labels_test=train_test_split(features, labels, test_size=.3,random_state=42)


# In[244]:

clf = clf.fit(features_train,labels_train)
pred =  clf.predict(features_test)

acc = accuracy_score(pred, labels_test)
print(acc)
#Using test_classifier from teter.py
test_classifier(clf, my_dataset, features_list, folds = 1000)


# The accuracy score from the sklearn.metrics is pretty high. In our case, because there are relatively small number of POIs in the dataset, accuracy is not very reliable metric. We need to look at precision and recall.
# 
# Precision is how many times the algorithm labeled items positive correctly, ie out of all items labeled positive how many truly belong to positive class.
# 
# Recall is how many times algorithm labels items as positive when they are truly positive.

# In[245]:

print classification_report(labels_test, pred)


# For POIs (row with label 1 above) we have precision 0.75, recall 0.6 and f1 score of 0.67.

# ## Question 6

# >Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

# Precision of 0.75 for POIs means that in 75% of times when algorithm labeled a person POI the person is actually a POI.
# 
# 
# Recall of 0.6 for POI means that 60% of actual POIs were correctly identified as POIs by the algorithm.
# 

# ## Dump the classifier, dataset, and features_list

# In[246]:

dump_classifier_and_data(clf, my_dataset, features_list)

