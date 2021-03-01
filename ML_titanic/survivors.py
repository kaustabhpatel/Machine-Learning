import pandas as pd
import numpy as np

train_data = pd.read_csv(open('train.csv'))
test_data = pd.read_csv(open('test.csv'))
test_labels_data = pd.read_csv(open('gender_submission.csv'))

features_list = []
for i in train_data:
    features_list.append(i)

### updating the age data which have NaN
### the average calculation has been done 
### through the whole train and test set 
male_age_sum = 0
male_count = 0
female_age_sum = 0 
female_count = 0

for i in range(len(train_data)):
    if not np.isnan(train_data['Age'][i]):
        if train_data['Sex'][i] == 'male':
            male_age_sum += train_data['Age'][i]
            male_count += 1
        else:
            female_age_sum += train_data['Age'][i]
            female_count += 1
        
for i in range(len(test_data)):
    if not np.isnan(test_data['Age'][i]):
        if test_data['Sex'][i] == 'male':
            male_age_sum += test_data['Age'][i]
            male_count += 1
        else:
            female_age_sum += test_data['Age'][i]
            female_count += 1

male_age_average = round(male_age_sum/male_count,2)
female_age_average = round(female_age_sum/female_count,2)

for i in range(len(train_data)):
    if np.isnan(train_data['Age'][i]):
        if train_data['Sex'][i] == 'male':
            train_data['Age'][i] = male_age_average
        else:
            train_data['Age'][i] = female_age_average

for i in range(len(test_data)):
    if np.isnan(test_data['Age'][i]):
        if test_data['Sex'][i] == 'male':
            test_data['Age'][i] = male_age_average
        else:
            test_data['Age'][i] = female_age_average

### selecting required features currently containing maximum features actually
### required according to me and can be reduced even further
required_features = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
    
### creating training and testing labels list type 
trainlabels = []
for i in train_data['Survived']:
    trainlabels.append(i)
    
testlabels = []
for i in test_labels_data['Survived']:
    testlabels.append(i)
    
### creating training and test features list type
trainfeatures = []
for i in range(len(train_data)):
    lists=[]
    for j in range(len(required_features)):
        lists.append(train_data[required_features[j]][i])
    trainfeatures.append(lists)

testfeatures = []
for i in range(len(test_data)):
    lists=[]
    for j in range(len(required_features)):
        lists.append(test_data[required_features[j]][i])
    testfeatures.append(lists)

### removing an unwanted data due to nan
### in fare in test_data
del testlabels[152]
del testfeatures[152]

### converting string values to numerical
### Sex, Embarked
for i in range(len(trainfeatures)):
    if trainfeatures[i][1] == 'male':
        trainfeatures[i][1] = 1
    else:
        trainfeatures[i][1] = 0
        
    """if trainfeatures[i][6] == 'C':
        trainfeatures[i][6] = 0
    elif trainfeatures[i][6] == 'Q':
        trainfeatures[i][6] = 1
    else:
        trainfeatures[i][6] = 2
"""
for i in range(len(testfeatures)):
    if testfeatures[i][1] == 'male':
        testfeatures[i][1] = 1
    else:
        testfeatures[i][1] = 0
        
    """if testfeatures[i][6] == 'C':
        testfeatures[i][6] = 0
    elif testfeatures[i][6] == 'Q':
        testfeatures[i][6] = 1
    else:
        testfeatures[i][6] = 2
"""
### creating numpy type labels and features
labels_train = np.array(trainlabels)
labels_test = np.array(testlabels)
features_train = np.array(trainfeatures)
features_test = np.array(testfeatures)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(pred,labels_test))