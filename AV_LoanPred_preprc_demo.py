
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd


# In[148]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()


# In[149]:


test.head()


# In[150]:


train.describe()


# In[60]:


train.info()


# In[151]:


train.info()
test.info()


# In[152]:


#missing value imputation
#No. of missing values in each column

train.isnull().sum()


# In[154]:


train['Married'].value_counts().index[0]


# In[155]:


#missing value imputation

from sklearn.preprocessing import Imputer
imp_median = Imputer(missing_values='NaN',strategy='median',axis=0,copy=False)
#imp_mode = Imputer(missing_values='NaN',strategy='most_frequent',axis=0,copy=False)
imp_median.fit_transform((train['LoanAmount']).reshape(-1, 1))
imp_median.fit_transform((train['Loan_Amount_Term']).reshape(-1, 1))

imp_median.fit_transform((test['LoanAmount']).reshape(-1, 1))
imp_median.fit_transform((test['Loan_Amount_Term']).reshape(-1, 1))


for column in ['Gender','Married','Dependents','Self_Employed','Credit_History']:
    mode = train[column].value_counts().index[0]
    train[column] = train[column].fillna(mode)

for column in ['Gender','Married','Dependents','Self_Employed','Credit_History']:
    mode = test[column].value_counts().index[0]
    test[column] = test[column].fillna(mode)


# In[157]:


test.isnull().sum()


# In[158]:


train.head()


# In[159]:


#converting string type columns to categories
for column in ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','Property_Area','Credit_History','Loan_Status']:
    train[column] = train[column].astype('category')

for column in ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']:
    test[column] = test[column].astype('category')


# In[160]:


test.info()


# In[161]:


#removing unneeded column 'Loan_ID'
train.drop('Loan_ID',axis=1,inplace=True)
IDs = test['Loan_ID']
test.drop('Loan_ID',axis=1,inplace=True)


# In[162]:


train.columns


# In[163]:


#Creating dummy variables for categorical attributes
cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Credit_History', 'Property_Area']
train1 = pd.get_dummies(train,prefix_sep="__",columns=cols,drop_first=True)
test1 = pd.get_dummies(test,prefix_sep="-",drop_first=True)


# In[164]:


train1.head()


# In[165]:


from sklearn import linear_model
model = linear_model.LogisticRegression(C=1e5)
X = train1.drop('Loan_Status',axis=1,inplace=False)
Y = train1['Loan_Status']
model.fit(X,Y)


# In[166]:


train_preds = model.predict(X)
test_preds = model.predict(test1)


# In[167]:


#confustion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confusion_matrix = confusion_matrix(Y, train_preds)
print(confusion_matrix)
print(classification_report(Y, train_preds))


# In[168]:


submission = pd.concat([IDs,pd.DataFrame(pd.Series(test_preds),columns=['Loan_Status'])],axis=1)
submission.to_csv('submission.csv',index=False)


# In[169]:


#rescaling and standardisation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']:
    train1[col] = scaler.fit_transform(train1[col].reshape(-1,1))
    test1[col] = scaler.fit_transform(test1[col].reshape(-1,1))
    


# In[170]:


train1.head()


# In[171]:


X = train1.drop('Loan_Status',axis=1,inplace=False)
Y = train1['Loan_Status']
model.fit(X,Y)
train_preds1 = model.predict(X)
test_preds1 = model.predict(test1)
print(classification_report(Y, train_preds1))


# In[172]:


#Standardise. useful technique to transform attributes with a Gaussian distribution
from sklearn.preprocessing import StandardScaler
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']:
    scaler = StandardScaler().fit(train1[col].reshape(-1,1))
    train1[col] = scaler.transform(train1[col].reshape(-1,1))
    scaler_test = StandardScaler().fit(test1[col].reshape(-1,1))
    test1[col] = scaler_test.transform(test1[col].reshape(-1,1))
train1.head()


# In[173]:


X = train1.drop('Loan_Status',axis=1,inplace=False)
Y = train1['Loan_Status']
model.fit(X,Y)
train_preds2 = model.predict(X)
test_preds2 = model.predict(test1)
print(classification_report(Y, train_preds2))


# In[175]:


submission = pd.concat([IDs,pd.DataFrame(pd.Series(test_preds2),columns=['Loan_Status'])],axis=1)
submission.to_csv('submission.csv',index=False)

