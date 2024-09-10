#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# ### Loading & processing the dataset

# In[11]:


data = pd.read_csv(r"E:\titanic\train.csv")


# In[12]:


data.head()


# In[13]:


data.shape


# In[15]:


# To check information of all columns
data.info()


# In[16]:


# To check null/ missing values
data.isna().sum()


# In[19]:


data = data.drop(columns='Cabin', axis=1)


# In[20]:


data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[21]:


data.info()


# In[22]:


data.isna().sum()


# In[23]:


data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)


# In[25]:


# To check duplicate value
data.duplicated().sum()


# ## Analysis the data

# In[28]:


data.describe()


# In[27]:


#To check How many survived  ## 1 means survived and 0 means not survived
data['Survived'].value_counts()


# In[29]:


data['Sex'].value_counts()


# In[41]:


sns.countplot(x='Sex',data=data)
plt.show()


# In[42]:


# Analysis Gender wise survived
sns.countplot(x='Sex',hue='Survived', data=data)
plt.show()


# In[43]:


# Analysis Class wise
sns.countplot(x='Pclass', data=data)
plt.show()


# In[44]:


sns.countplot(x='Pclass', hue= 'Survived', data=data)


# In[45]:


# Encoding 
from sklearn.preprocessing import LabelEncoder


# In[46]:


le=LabelEncoder()


# In[51]:


data['Sex'] = le.fit_transform(data['Sex'])


# In[53]:


data['Embarked'] =le.fit_transform(data['Sex'])


# In[54]:





# In[55]:


data.columns


# In[56]:


x = data.drop(columns={'PassengerId','Survived','Name','Ticket'},axis=1)
y = data['Survived']


# In[57]:


x


# In[58]:


y


# In[59]:


# Training the data
from sklearn.model_selection import train_test_split


# In[72]:


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=13)


# In[73]:


x_train


# ## Regression model 

# In[74]:


from sklearn.linear_model import LogisticRegression


# In[75]:


lr =LogisticRegression()


# In[76]:


lr.fit(x_train,y_train)


# In[70]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import *


# In[77]:


# Train Data prediction
x_train_pred=lr.predict(x_train)


# In[78]:


accuracy_score(y_train,x_train_pred)


# In[79]:


#Test data prediction

x_test_pred=lr.predict(x_test)


# In[80]:


accuracy_score(y_test,x_test_pred)


# ## Build predictive model

# In[85]:


# Here input the x_train features
input_data = (3,1,30,0,0,200,1)

# change the input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

#Reshape the numpy array 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = lr.predict(input_data_reshaped)

print(prediction)


if (prediction[0]==0):
    print('The person does not survived')
else:
    print('The person survived')
    


# In[ ]:




