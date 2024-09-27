#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Dataset

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[17]:


data=pd.read_csv ("C:/Users/Fatlem/Desktop/Data-Mining/Fatlem-Rep1/heart.csv")
data


# ## Display the first few rows of the dataset and summary information

# In[14]:


heart_data_info = heart_data.info()
heart_data_head = heart_data.head()

heart_data_info, heart_data_head


# In[7]:


X = heart_data.drop('target', axis=1)
y = heart_data['target']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[9]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Initialize and train the Random Forest model

# In[10]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


# ## Initialize and train the Random Forest model

# In[11]:


y_pred = model.predict(X_test_scaled)


# ## Evaluate the model

# In[12]:


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)


# ## Display the results

# In[13]:


print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_mat)

