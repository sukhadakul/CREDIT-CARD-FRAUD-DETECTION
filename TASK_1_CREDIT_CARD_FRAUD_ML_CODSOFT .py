#!/usr/bin/env python
# coding: utf-8

# # TASK 1 Machine Learning Intern @Codsoft

# # CREDIT CARD FRAUD DETECTION 

# 
# Build a model to detect fraudulent credit card transactions. Use a dataset containing information about credit card transactions, and experiment with algorithms like Logistic Regression, Decision Trees, or Random Forests to classify transactions as fraudulent or legitimate.

# # 1) Import Dependencies 📦📚 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sn 
import sklearn.svm as svm 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# # 2) Load Dataset & Data Preprocessing 🔄🔢 

# In[2]:


Data= pd.read_csv("credit_card_Dataset.csv") 
Data  


# In[3]:


#check our data shape 
Data.shape 


# In[4]:


#check our data types 
Data.dtypes 


# In[5]:


Data.info() 


# In[6]:


Data.describe() 


# In[7]:


# Check Null Values
Data.isnull().sum()    


# No null value in our dataset 

# In[8]:


# Check Duplicate Values
Data.duplicated().sum()   


# In[9]:


Data.drop_duplicates() 


# In[10]:


Data['Class'].value_counts() 
 


# In[11]:


# Drop non-numeric columns (e.g., 'Date' or 'Timestamp')
Data = Data.select_dtypes(include=[np.number]) 


# In[12]:


# Create the correlation matrix
correlation_matrix = Data.corr()
correlation_matrix 


# # 3) (EDA)📊 Exploratory Data Analysis 📊

# # Histograms📊 

# In[13]:


Data.hist(bins=30,figsize=(30,15)) 


# # Heatmap 🌡️ 

# In[14]:


# Create a heatmap
plt.figure(figsize=(20, 10))
sn.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show() 


# In[15]:


# Visualize the class distribution
plt.figure(figsize=(6, 4))
sn.countplot(data=Data, x='Class')
plt.xlabel('Class')
plt.ylabel('count')
plt.title('Count of Fraudulent and Non-fraudulent Transactions')
plt.show() 


# # 4) Models Training / Building 📈

# In[16]:


# Data preprocessing
X = Data.drop('Class', axis=1)
y = Data['Class'] 


# In[17]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


# In[18]:


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 


# In[19]:


# Train a Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train) 


# In[20]:


# Train a Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train) 


# In[21]:


# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)  


# In[22]:


# Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) 
    
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1) 


# In[23]:


print("Logistic Regression Model:")
evaluate_model(lr_model, X_test, y_test) 


# In[24]:


print("Decision Tree Model:")
evaluate_model(dt_model, X_test, y_test) 


# In[25]:


print("Random Forest Model:")
evaluate_model(rf_model, X_test, y_test)  


# # 5) Accuracy 🎯💯🚀 

# # I got a Accuracy Score: 0.99 (99%)
