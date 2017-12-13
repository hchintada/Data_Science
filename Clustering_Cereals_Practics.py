
# coding: utf-8

# In[62]:


#Importing required modules and setting the working directory
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from matplotlib import pyplot as plt
os.chdir('C:\\Users\\virinchi\\Downloads')


# In[34]:


#Loading the dataset into Python
data = pd.read_csv('Cereals.csv')


# In[ ]:


#A look at the dataset
print(data.head())
print(data.info())


# In[36]:


#Checking the first column
print(len(data['name'].unique()))

#removing the first column as it doesn't add extra value
data.drop(['name'],axis=1,inplace=True)


# In[73]:


#checking missing values
print(data.isnull().sum())


# In[40]:


#standardising the dataset as clustering techniques use distance metric
from sklearn.preprocessing import StandardScaler
for col in data.columns:
    scaler = StandardScaler().fit(data[col].values.reshape(-1,1))
    data[col] = scaler.transform(data[col].values.reshape(-1,1))
    
data.head()


# In[43]:


#Fitting the Kmeans clustering model
KM_model = KMeans(n_clusters=4)
KM_model.fit(data)


# In[47]:


#A look at the clustered labels and centroids
labels = KM_model.labels_
cluster_centers = KM_model.cluster_centers_


# In[51]:


labels


# In[52]:


#Fitting an agglomerativeclustering model
Ag_model = AgglomerativeClustering(n_clusters = 4)
Ag_model.fit(data)


# In[53]:


Ag_model.labels_


# In[56]:


#visualising the dendogram
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# In[71]:


from scipy.cluster.hierarchy import dendrogram, linkage
plt.figure(figsize=(20,10))
plot_dendrogram(Ag_model,labels=Ag_model.labels_)
plt.show()

