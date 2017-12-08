
# coding: utf-8

# In[2]:


import pandas as pd
import os
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[5]:


os.chdir('C:\\Users\\virinchi\\Downloads')


# In[6]:


data = pd.read_csv('CustomerData.csv')

data['City'] = data['City'].astype('category')
data.drop(['CustomerID'],axis=1,inplace=True)


# In[11]:


data1 = pd.get_dummies(data,columns=['FavoriteChannelOfTransaction','FavoriteGame'],drop_first=True)

X=data1.drop(['Revenue'],axis=1)
Y=data1['Revenue']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


# In[13]:


model = linear_model.LinearRegression()
model.fit(x_train,y_train)
train_preds = model.predict(x_train)
tests = model.predict(x_test)


# In[1]:


X.head()


# In[115]:


mse = mean_squared_error(y_train,train_preds)
mse_t = mean_squared_error(y_test,tests)
print(mse)
print(mse_t)


# In[ ]:


from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=6)







# In[116]:



knn.fit(x_train,y_train)
knn_train_preds = knn.predict(x_train)
knn_test_preds = knn.predict(x_test)
mseKnn = mean_squared_error(y_train,knn_train_preds)
mse_tKnn = mean_squared_error(y_test,knn_test_preds)
print(mseKnn)
print(mse_tKnn)


# In[10]:


data.head()


# In[147]:


#standardising and checking how it influences error

Xk=data.drop(['TotalRevenueGenerated'],axis=1)
Yk=data['TotalRevenueGenerated']
xk_train,xk_test,yk_train,yk_test = train_test_split(Xk,Yk,test_size=0.3,random_state=0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#cols = list(filter(lambda col: (str(x_train[col].dtype) != 'category'), cols))
num_cols = [col for col in xk_train.columns if str(xk_train[col].dtype) != 'category' and xk_train[col].dtype != 'O']

for col in num_cols:
    xk_train.loc[:,col] = scaler.fit_transform(xk_train[col].values.reshape(-1, 1))
    xk_test.loc[:,col] = scaler.fit_transform(xk_test[col].values.reshape(-1, 1))


# In[148]:


#dummifying categorical variables
cat_cols = [col for col in xk_train.columns if str(xk_train[col].dtype) == 'category' or xk_train[col].dtype == 'O']
xk_train = pd.get_dummies(xk_train,columns=cat_cols,drop_first=True,prefix_sep = '_')
xk_test = pd.get_dummies(xk_test,columns=cat_cols,drop_first=True,prefix_sep = '_')


# In[188]:


#fitting KNN and predicting on it
knn2 = neighbors.KNeighborsRegressor(n_neighbors=6)
knn2.fit(xk_train,yk_train)
knn2_train_preds = knn2.predict(xk_train)
knn2_test_preds = knn2.predict(xk_test)
mseKnn2 = mean_squared_error(yk_train,knn2_train_preds)
mse_tKnn2 = mean_squared_error(yk_test,knn2_test_preds)
print(mseKnn2)
print(mse_tKnn2)

