#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis - Retail
# Author - Vinay Kumar

# In[2]:


pip install seaborn


# In[4]:


pip install sklearn


# In[21]:


from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv(r'C:\Users\n\Desktop\Dataset\Iris.csv')  #loading dataset
df.head()    #display starting 3 rows


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


df.drop_duplicates(inplace=True)


# Label Encoding

# In[12]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])
df['Species'].value_counts()


# # PetalLengthCm vs PetalWidthCm
# Will use in end to  compare our final plot with this graph to check how accurate our model is

# In[13]:


plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'],c=df.Species.values)


# In[14]:


df.corr()


# # Data Visualization

# In[22]:


fig=plt.figure(figsize=(15,12))
sns.heatmap(df.corr(),linewidths=1,annot=True)


# In[23]:


sns.pairplot(df)


# From this we can say that Species is mainly depend on Petal Length and Petal Width.
# 
# Using Petal_length and Petal_width

# In[24]:


df=df.iloc[:,[0,1,2,3]].values


# Elbow Method using within-cluster-sum-of-squares(wcss)

# In[26]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
wcss


# Using Elbow graph to find optimum no. of Clusters

# In[27]:


plt.figure(figsize=(10,5))
sns.set(style='whitegrid')
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# The optimum value for K would be 3. 

# As we can see that with an increase in the number of clusters the WCSS value decreases. 

# We select the value for K on the basis of the rate of decrease in WCSS and we can see that after 3 the drop in wcss is minimal.

# # Initialization using K-means++

# In[28]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 5)
y_kmeans = kmeans.fit_predict(df)
y_kmeans


# # Visualizing the Clusters

# In[30]:


fig = plt.figure(figsize=(10, 7))
plt.title('Clusters with Centroids',fontweight ='bold', fontsize=20)
plt.scatter(df[y_kmeans == 0, 2], df[y_kmeans == 0, 3], s = 100, c = 'seagreen', label = 'Iris-versicolour')
plt.scatter(df[y_kmeans == 1, 2], df[y_kmeans == 1, 3], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(df[y_kmeans == 2, 2], df[y_kmeans == 2, 3],s = 100, c = 'yellow', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s = 300, c = 'red',marker='*', 
            label = 'Centroids')
plt.title('Iris Flower Clusters')
plt.ylabel('Petal Width in cm')
plt.xlabel('Petal Length in cm')
plt.legend()


# # Our predicted graph is quite similar to the Actual One.
