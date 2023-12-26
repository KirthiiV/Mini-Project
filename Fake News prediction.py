#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import seaborn as sns                               
import matplotlib.pyplot as plt


# In[3]:


df =pd.read_csv('train.csv.zip')


# In[4]:


df.head(5)


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df=df.fillna('')


# In[9]:


df.isnull().sum()


# In[10]:


df.columns


# In[11]:


df = df.drop(['id', 'title', 'author'], axis = 1)


# In[12]:


df.head()


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[14]:


#Create a function to process the text
def wordopt(text):
  text =text.lower()
  text = re.sub('\[.*?\]','', text)
  text = re.sub("\\W"," ", text)
  text = re.sub('https?://\S+|www\.\S+','', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]'% re.escape(string.punctuation), '', text)
  text = re.sub('\n', '',text)
  text = re.sub('\w*\d\w*', '', text)
  return text

df['text'] = df['text'].apply(wordopt)


# In[15]:


#Dependent and independent variables
x = df['text']
y= df['label']

#splitting training and testing datas:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


# In[16]:


#convert text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[27]:


vect = TfidfVectorizer()


# In[ ]:





# In[30]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[31]:


import pickle


# In[32]:


pickle.dump(vect, open('vector.pkl', 'wb'))


# In[33]:


pickle.dump(LR, open('LR.pkl', 'wb'))


# In[34]:


vector_form=pickle.load(open('vector.pkl', 'rb'))


# In[36]:


load_model = pickle.load(open('LR.pkl', 'rb'))


# In[37]:


pred_lr=LR.predict(xv_test)

LR.score(xv_test, y_test)


# In[38]:


print(classification_report(y_test, pred_lr))


# In[39]:


#Model testing
def output_lable(n):
    if n == 0:
        return "True News"
    elif n == 1:
        return "Fake News"

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    return print("\n\nLR Prediction:{}".format(output_lable(pred_LR[0])))


# In[25]:


news = str(input())
manual_testing(news)


# In[ ]:




