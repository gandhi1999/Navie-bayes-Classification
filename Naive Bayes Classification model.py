#!/usr/bin/env python
# coding: utf-8

# Naive Bayes is one of the classification model. we can apply this model to any classification model. but this model will suit for the target column should be categorial but it should have continuoes in nature in this case it will give better performance.

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[46]:


df=pd.read_csv("C:/Users/USER/Desktop/Machine Learnng/Ml_bi_data/TrumpTrudeau.csv")


# In[47]:


df


# In[48]:


df.isnull().sum()


# In[49]:


target=df["author"]


# In[50]:


feature=df["status"]


# In[51]:


feature


# In[52]:


from sklearn.feature_extraction.text import CountVectorizer


# In[53]:


vectorizer=CountVectorizer(stop_words="english")
feature=vectorizer.fit_transform(feature)


# In[54]:


feature.shape


# In[55]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(feature,target,test_size=0.2,random_state=3,stratify=target)


# In[56]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)


# In[57]:


model.score(x_test,y_test)


# In[58]:


tweet=["canada"]


# In[59]:


tweet=vectorizer.transform(tweet)


# In[60]:


model.predict(tweet)


# In[ ]:




