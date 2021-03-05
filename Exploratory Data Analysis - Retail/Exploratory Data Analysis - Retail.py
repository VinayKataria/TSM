#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis - Retail 

# Author - Vinay Kumar

# In[1]:


pip install seaborn


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[11]:


df = pd.read_csv(r'C:\Users\n\Desktop\Dataset\Retail(Dataset).csv')  #loading dataset
df.head()    #display starting 3 rows


# In[13]:


df.tail() #display ending 3 rows


# In[14]:


df.shape


# In[15]:


df.describe()       #Summary


# In[16]:


df.isnull().sum()       #Null value checking


# In[17]:


df.info()           #info regarding dataset


# In[18]:


df.columns


# In[19]:


df.duplicated().sum()


# In[20]:


df['Postal Code'] = df['Postal Code'].astype('object')


# In[21]:


df.drop_duplicates(subset=None,keep='first',inplace=True)
df.duplicated().sum()


# In[26]:


corr = df.corr()
sns.heatmap(corr,annot=True,cmap='Blues')


# In[27]:


df = df.drop(['Postal Code'],axis = 1)    #dropping postal code columns


# In[32]:


sns.pairplot(df, hue = 'Ship Mode')


# In[33]:


df['Ship Mode'].value_counts()


# In[34]:


sns.countplot(x=df['Ship Mode'])


# In[35]:


df['Segment'].value_counts()        #Segment ValueCount


# In[36]:


sns.pairplot(df,hue = 'Segment')     #Plot Pair Plot


# In[37]:


sns.countplot(x = 'Segment',data = df, palette = 'rainbow')


# In[38]:


df['Category'].value_counts()  #Category ValueCount


# In[39]:


sns.countplot(x='Category',data=df,palette='tab10')


# In[40]:


sns.pairplot(df,hue='Category')  #Plot Pair Plot


# In[41]:


df['Sub-Category'].value_counts() #Sub-Category ValueCount


# In[42]:


plt.figure(figsize=(15,12))
df['Sub-Category'].value_counts().plot.pie(autopct='dark')
plt.show()


# # Observation 1

# In[43]:


df['State'].value_counts()


# In[44]:


plt.figure(figsize=(15,12))
sns.countplot(x='State',data=df,palette='rocket_r',order=df['State'].value_counts().index)
plt.xticks(rotation=90)
plt.show()


# # Observation 2

# In[45]:



df.hist(figsize=(10,10),bins=50)
plt.show()


# # Observation 3

# In[47]:



plt.figure(figsize=(10,8))
df['Region'].value_counts().plot.pie(autopct = '%1.1f%%')
plt.show()


# # Profit vs Discount

# In[48]:


fig,ax=plt.subplots(figsize=(20,8))
ax.scatter(df['Sales'],df['Profit'])
ax.set_xlabel('Sales')
ax.set_ylabel('Profit')
plt.show()


# In[49]:


sns.lineplot(x='Discount',y='Profit',label='Profit',data=df)
plt.legend()
plt.show()


# # Observation 4
# No correlation between profit and discount
# 

# # Profit vs Quantity

# In[50]:


sns.lineplot(x='Quantity',y='Profit',label='Profit',data=df)
plt.legend()
plt.show()


# In[68]:


df.groupby('Segment')[['Profit','Sales']].sum().plot.bar(color=['Cyan','Pink'],figsize=(8,5))
plt.ylabel('Profit/Loss and Sales')
plt.show()


# # Observation 5
# Profit and sales are maximum in consumer segment and minimum in Home Office segment

# In[69]:



plt.figure(figsize=(12,8))
plt.title('Segment wise Sales in each Region')
sns.barplot(x='Region',y='Sales',data=df,hue='Segment',order=df['Region'].value_counts().index,palette='rocket')
plt.xlabel('Region',fontsize=15)
plt.show()


# # Observation 6
# Segment wise sales are almost same in every region

# In[71]:



df.groupby('Region')[['Profit','Sales']].sum().plot.bar(color=['Cyan','Pink'],figsize=(8,5))
plt.ylabel('Profit/Loss and sales')
plt.show()


# # Observation 7
# Profit and sales are Maximum in West Region and Minimum in South Region

# In[78]:


ps = df.groupby('State')[['Sales','Profit']].sum().sort_values(by='Sales',ascending=False)
ps[:].plot.bar(color=['Cyan','pink'],figsize=(15,8))
plt.title('Profit/loss & Sales across States')
plt.xlabel('States')
plt.ylabel('Profit/loss & Sales')
plt.show()


# # Observation 8
# High Profit is for California, New York And Loss is for Texas, Pennsylvania, Ohio

# In[79]:



t_states = df['State'].value_counts().nlargest(10)
t_states


# In[80]:


df.groupby('Category')[['Profit','Sales']].sum().plot.bar(color=['Cyan','pink'],alpha=0.9,figsize=(8,5))
plt.ylabel('Profit/Loss and sales')
plt.show()


# # Observation 9
# As a business manager, try to find out the weak areas where you can work to make more profit?
# 

# Technology and Office Supplies have high profit.
# 

# Furniture have less profit

# In[82]:



ps = df.groupby('Sub-Category')[['Sales','Profit']].sum().sort_values(by='Sales',ascending=False)
ps[:].plot.bar(color=['Cyan','pink'],figsize=(15,8))
plt.title('Profit/loss & Sales across states')
plt.xlabel('Sub-Category')
plt.ylabel('Profit/loss & Sales')
plt.show()


# # Observation 10
# Phones sub-category have high sales.
# 

# Chairs have high sales but less profit compared to phones
# 

# Tables and Bookmarks sub-categories facing huge loss
