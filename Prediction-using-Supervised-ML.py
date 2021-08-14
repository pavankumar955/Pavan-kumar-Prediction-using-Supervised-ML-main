#!/usr/bin/env python
# coding: utf-8

# # SONALI PATIL
# # Task 1 - Prediction using Supervised ML (Level - Beginner)
# 

# In[25]:


# import all reuired libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


# Reading the file 
url="http://bit.ly/w-data"
data=pd.read_csv(url)


# In[27]:


data.head()


# In[28]:


data.info()


# In[29]:


data.describe()


# # Visualizing the data
# 

# In[30]:


## Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='*')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # From above graph we can clearly see there is a positive linear relation bet. no. of hours and percentage of the score

# # Preparing the data 

# In[31]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[46]:


# Split this data into a training and test set
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[48]:


print("Shape of X_train",X_train.shape)
print("shape of y_train",y_train.shape)
print("Shape of X_test",X_test.shape)
print("Shape of y_test",y_test.shape)


# In[50]:


## After the spliting  now we have to train our algorithm

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete!!!!!.")


# # Plotting Regression Line

# In[56]:


regressor.coef_


# In[57]:


regressor.intercept_


# In[58]:


line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.plot(X, line);
plt.show()


# # Predictions

# In[36]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[37]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[59]:


# Predict the value by own data
hours = [9.25]
own_pred = regressor.predict([hours])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Evaluting the model

# #The final step is to evaluate the performance of algorithm.
# #This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity #here, I have evaluted  model using mean absolute error,mean squared error and root mean squared error
# 

# In[63]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




