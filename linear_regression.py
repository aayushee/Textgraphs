from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from math import sin
import numpy as np
import csv
import pandas as pd
from sklearn.utils import resample
import sys

number_of_features=sys.argv[1]
mode='train'
fname='questions/ilp_'+str(number_of_features)+'scores_'+mode+'.csv'

df=pd.read_csv(fname)
# Separate majority and minority classes
df_majority = df[df.Label==0]
df_minority = df[df.Label==1]

df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_minority),     # to match minority class
                                 random_state=123) # reproducible results

df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled.to_csv('questions/ilp_'+str(number_of_features)+'scores_resampled_'+mode+'.csv',index=False)


fname='questions/ilp_'+str(number_of_features)+'scores_resampled_'+mode+'.csv'

Data = np.genfromtxt(fname, delimiter=',')[1:] # Remove the CSV header


col1=3
if number_of_features=='2':
    col2=5
if number_of_features=='3':
    col2=6 
if number_of_features=='4':
    col2=7

labels = Data[:,col2]
features = Data[:,col1:col2]

features_scaled = preprocessing.scale(features)

# Train model
model = LinearRegression()
model.fit(features_scaled, labels)
intercept=model.intercept_

print("coef of determination: ",model.coef_)
print ("intercept value: ",intercept)

mode='test'
fname_test='questions/ilp_'+str(number_of_features)+'scores_'+mode+'.csv'
scaledscore=[]

TestData = np.genfromtxt(fname_test, delimiter=',')[1:] # Remove the CSV header

#print(labels[0:10])
test_features = TestData[:,2:]
#print(features[0:10])
test_features_scaled=preprocessing.scale(test_features)
print(test_features_scaled.shape[0])
#print(test_features_scaled[0])

def relevanceScore4(intercept, f1Coef, f2Coef, f3Coef, f4Coef,qpa, paa, ipa, isa):
    return intercept + (f1Coef * qpa) + (f2Coef * paa) + (f3Coef * ipa) + (f4Coef * isa)


def relevanceScore3(intercept, f1Coef, f2Coef, f3Coef ,qpa, paa, ipa):
    return intercept + (f1Coef * qpa) + (f2Coef * paa) + (f3Coef * ipa) 

def relevanceScore2(intercept, f1Coef, f2Coef, qpa, paa):
    return intercept + (f1Coef * qpa) + (f2Coef * paa)

f1Coef=model.coef_[0]
f2Coef=model.coef_[1]    
for i in range (0,test_features_scaled.shape[0]):
    qpa=test_features_scaled[i][0]
    paa=test_features_scaled[i][1]
    if len(model.coef_)==4:
        f3Coef=model.coef_[2]
        f4Coef=model.coef_[3]
        ipa=test_features_scaled[i][2]
        isa=test_features_scaled[i][3]
        scaledscore.append(relevanceScore4(intercept,f1Coef,f2Coef,f3Coef,f4Coef,qpa,paa,ipa,isa))
    elif len(model.coef_)==3:
        f3Coef=model.coef_[2]
        ipa=test_features_scaled[i][2]
        scaledscore.append(relevanceScore3(intercept,f1Coef,f2Coef,f3Coef,qpa,paa,ipa))
    else:
        scaledscore.append(relevanceScore2(intercept,f1Coef,f2Coef,qpa,paa))

 

df_test = pd.read_csv(fname_test)
 
df_test['CombinedScoreScaled']=scaledscore

df_test.to_csv(fname_test,index=False)
    
