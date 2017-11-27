
# coding: utf-8

# # Logistic Regression

# In[63]:

import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')



# In[64]:

iris = datasets.load_iris()
X = iris.data[:,:2] #Choosing only the first two input-features
Y = iris.target
#The first 50 samples are class 0 and the next 50 samples are class 1
X = X[:100]
Y = Y[:100]


# In[65]:

###Splitting into training and test sets###
number_of_samples = len(Y)

random_indices = np.random.permutation(number_of_samples,) #shuffling numbers to selected indices for train and test
#Training set
num_training_samples = int(number_of_samples*0.8)
x_train = X[random_indices[:num_training_samples]]#selecting first 80 rows for train set (recall list/np.array slicing list[:5])
y_train = Y[random_indices[:num_training_samples]]
#Test set
num_test_samples = int(number_of_samples*0.2)
x_test = X[random_indices[-num_test_samples:]]#selecting the last 20 rows for train set (recall list/np.array slicing list[-2:])
y_test = Y[random_indices[-num_test_samples:]]


# In[72]:

#For Demonstrating a two-class classification task we pick only the 0 and 1 classes
X_class0 = np.asmatrix([x_train[i] for i in range(len(x_train)) if y_train[i]==0]) #Picking only the first two classes
Y_class0 = np.zeros((X_class0.shape[0]),dtype=np.int)
X_class1 = np.asmatrix([x_train[i] for i in range(len(x_train)) if y_train[i]==1])
Y_class1 = np.ones((X_class1.shape[0]),dtype=np.int)

full_X = np.concatenate((X_class0,X_class1),axis=0)
full_Y = np.concatenate((Y_class0,Y_class1),axis=0)

X_class0_test = np.asmatrix([x_test[i] for i in range(len(x_test)) if y_test[i]==0]) #Picking only the first two classes
Y_class0_test = np.zeros((X_class0_test.shape[0]),dtype=np.int)
X_class1_test = np.asmatrix([x_test[i] for i in range(len(x_test)) if y_test[i]==1])
Y_class1_test = np.ones((X_class1_test.shape[0]),dtype=np.int)

full_X_test = np.concatenate((X_class0_test,X_class1_test),axis=0)
full_Y_test = np.concatenate((Y_class0_test,Y_class1_test),axis=0)


# In[67]:

#Visualizing the training data
X_class0 = np.asmatrix([x_train[i] for i in range(len(x_train)) if y_train[i]==0]) #Picking only the first two classes
Y_class0 = np.zeros((X_class0.shape[0]),dtype=np.int)
X_class1 = np.asmatrix([x_train[i] for i in range(len(x_train)) if y_train[i]==1])
Y_class1 = np.ones((X_class1.shape[0]),dtype=np.int)

plt.scatter([X_class0[:,0]], [X_class0[:,1]],color='red')
plt.scatter([X_class1[:,0]], [X_class1[:,1]],color='blue')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['class 0','class 1'])
plt.title('Fig 3: Visualization of training data')
plt.show()


# In[49]:

#fitting the logreg model

model = linear_model.LogisticRegression(C=1e5)#C is the inverse of the regularization factor
model.fit(full_X,full_Y)


# In[51]:

# Display the decision boundary
#(Visualization code taken from: http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].

h = .02  # step size in the mesh
x_min, x_max = full_X[:, 0].min() - .5, full_X[:, 0].max() + .5
y_min, y_max = full_X[:, 1].min() - .5, full_X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #predict for the entire mesh to find the regions for each class in the feature space

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter([X_class0[:, 0]], [X_class0[:, 1]], c='red', edgecolors='k', cmap=plt.cm.Paired)
plt.scatter([X_class1[:, 0]], [X_class1[:, 1]], c='blue', edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Fig 4: Visualization of decision boundary')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())


plt.show()


# In[74]:

#Predicting on the test set.
test_preds = model.predict(full_X_test)


# In[75]:

print(full_Y_test)
print(test_preds)


# In[76]:

#confustion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(full_Y_test, test_preds)
confusion_matrix


# In[77]:

#Classification metrics report
print(classification_report(full_Y_test, test_preds))

