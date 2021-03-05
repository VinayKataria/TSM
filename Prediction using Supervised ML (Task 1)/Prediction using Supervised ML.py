#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML
# Author: Vinay Kumar

# In[1]:


#Libraies Used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import seaborn as sns


# In[2]:


#Inputing the Data from provided link
data = pd.read_csv('http://bit.ly/w-data')
data.head(5)


# In[3]:


# Null Value checkup
data.isnull == True


# In[4]:


sns.set_style('white')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Obtained Vs Studied Hours',size=25)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Studied Hours', size=12)
plt.show()


# In[5]:


#Regression line to confirm the correlation
sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Studied Hours', size=12)
plt.show()
print(data.corr())


# # Variables are Positively Correlated

# # Model Training

# In[6]:


#Data Spliting
# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values
# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[7]:


#Fitting the Data into the model
regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# In[8]:


#Predicting the Percentage of Marks
pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# In[9]:


#Comparing the Predicted Marks with the Actual Marks
compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[10]:


#Visually Comparing the Predicted Marks with the Actual Marks
plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[11]:


#Evaluating the Model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# # What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?

# In[12]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.
