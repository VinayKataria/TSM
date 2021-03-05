#!/usr/bin/env python
# coding: utf-8

# # Prediction using Decision Tree Algorithm
# Author Vinay Kumar

# In[1]:


import sklearn.datasets as datasets
import pandas as pd


# In[2]:


iris=datasets.load_iris()


# In[3]:


X = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[4]:


X.head()


# In[5]:


X.tail()


# In[6]:


X.info()


# In[7]:


X.describe()


# In[8]:


X.isnull().sum()


# In[9]:


Y = iris.target
Y


# Split dataset into train and test sets

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# Defining the Decision Tree Algorithm

# In[11]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
print('Decision Tree Classifer Created Successfully')


# In[12]:


y_predict = dtc.predict(X_test)


# Constructing confusion matrix

# In[13]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)


# In[14]:


from sklearn import tree
import matplotlib.pyplot as plt


# Visualizing the Decision tree

# In[15]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa','versicolor','virginica']
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)
tree.plot_tree(dtc, feature_names = fn, class_names = cn, filled = True);


# You can now feed any new/test data to this classifer and it would be able to predict the right class accordingly.
