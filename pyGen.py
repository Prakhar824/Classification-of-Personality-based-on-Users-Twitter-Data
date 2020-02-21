import csv
import array
import pandas
import pickle
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import svm
csvFile=open('newfrequency300.csv', 'rt')
csvReader=csv.reader(csvFile)
mydict={row[1]: int(row[0]) for row in csvReader}

y=[]
with open ('PJFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('PJFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
#model=GaussianNB()
model=MultinomialNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
print('Training model for Judging/Perception')
print(train.shape)
print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
print(model)
print('Accuracy : %f' % accuracy_score(y_true=xtrain[0][:66403],y_pred=xtest[0][:]))
pickle.dump(model, open('BNPJFinal.txt', 'wb'))
del result

y=[]
with open ('IEFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('IEFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
#model=GaussianNB()
model=MultinomialNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
print('Training model for Introversion/Extraversion')
print(train.shape)
print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
print(model)
print('Accuracy : %f' % accuracy_score(y_true=xtrain[0][:85570],y_pred=xtest[0][:]))
pickle.dump(model, open('BNIEFinal.txt', 'wb'))
del result

y=[]
with open ('TFFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('TFFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
#model=GaussianNB()
model=MultinomialNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
print('Training model for Thinking/Feeling')
print(train.shape)
print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
print(model)
print('Accuracy : %f' % accuracy_score(y_true=xtrain[0][:64000],y_pred=xtest[0][:]))
pickle.dump(model, open('BNTFFinal.txt', 'wb'))
del result

y=[]
with open ('SNFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('SNFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
#model=GaussianNB()
model=MultinomialNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
print('Training model for Sensing/iNtuition')
print(train.shape)
print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
print(model)
print('Accuracy : %f' % accuracy_score(y_true=xtrain[0][:47135],y_pred=xtest[0][:]))
pickle.dump(model, open('BNSNFinal.txt', 'wb'))
