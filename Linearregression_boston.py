from sklearn import datasets  ## imports datasets from scikit-learn
data = datasets.load_boston()  ## loads Boston dataset from datasets library

#print(data.DESCR)  # DESCR works only for sklearn datasets

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

# define the data/predictors as the pre-set feature names
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
print(df.head())
#print(df.describe())
#print(target.head())

#splitting the data into train and test sets to cross check model performance on unseen data
x_train, x_test, y_train, y_test = train_test_split(df,target,test_size=0.2,random_state=4)

lm = linear_model.LinearRegression() #creating a model object
model = lm.fit(x_train,y_train) #fitting the linear reg model on the data

train_preds = lm.predict(x_train) #predicting on the train data
print(train_preds[0:5])
print(y_train[:5])

lm.score(x_train,y_train) #R squared value
lm.coef_ #gives the coefficients of the predictors
lm.intercept_ #y-intercept

from sklearn.metrics import mean_squared_error

meanSquaredError=mean_squared_error(y_train, train_preds)
print("Mean squared error for the train set is", meanSquaredError)

#predicting on unseen data
test_preds = lm.predict(x_test)
print(test_preds[0:5])
print(y_test[:5])

meanSquaredError_test=mean_squared_error(y_test, test_preds)
print("Mean squared error for the test set is", meanSquaredError_test)


#Another library for regression
import statsmodels.api as sm

model1 = sm.OLS(y_train,x_train).fit()
train_preds1 = model1.predict(x_train)
print(model1.summary())


#Residual Plots
import matplotlib.pyplot as plt
plt.scatter(train_preds, train_preds-y_train, c='g', s=40, alpha=0.5)
plt.scatter(test_preds, test_preds-y_test, c='r', s=40, alpha=0.5)
plt.hlines(y=0,xmin=0,xmax=50)
plt.title('Residual Plot (train in blue and test in red)')
plt.ylabel('Residual')
plt.show()