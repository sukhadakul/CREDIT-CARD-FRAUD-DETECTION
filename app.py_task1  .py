#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install streamlit')


# In[2]:


import streamlit as st
import requests

st.title("Credit Card Fraud Detection")

# Create input widgets for user input
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
# Add more input widgets as needed

if st.button("Predict Fraud"):
    user_input = [feature1, feature2]  # Create a list of user input features
    data = {'data': user_input}
    response = requests.post("http://localhost:5000/predict", json=data)
    
    if response.status_code == 200:
        prediction = response.json()[0]
        st.write("Predicted Class:", prediction)
    else:
        st.write("Prediction failed")

# Additional Streamlit components for displaying data and results
# (You can add more visualizations here)


# In[ ]:




