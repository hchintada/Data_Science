
# coding: utf-8

# In[4]:


#Decision Trees
import numpy as np
from sklearn import datasets, tree
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

iris = datasets.load_iris()
X = iris.data #Choosing only the first two input-features
Y = iris.target

number_of_samples = len(Y)
#Splitting into training, validation and test sets
random_indices = np.random.permutation(number_of_samples)
#Training set
num_training_samples = int(number_of_samples*0.8)
x_train = X[random_indices[:num_training_samples]]
y_train = Y[random_indices[:num_training_samples]]
#Test set
num_test_samples = int(number_of_samples*0.2)
x_test = X[random_indices[-num_test_samples:]]
y_test = Y[random_indices[-num_test_samples:]]


# In[5]:


#fitting the model

model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[6]:


#visualising the decision tree

from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image

dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data,  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())


# In[11]:


#Predicting on the test set.
test_preds = model.predict(x_test)

#confustion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y_test, test_preds))

#Classification metrics report
print(classification_report(y_test, test_preds))

