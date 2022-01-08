#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('load_ext', 'autotime')


# In[2]:


data = pd.read_csv('suspicious tweets.csv')
data.head()


# In[3]:


data.groupby('label').describe()


# In[4]:


pd.value_counts(data['label'])


# In[5]:


plt.pie(pd.value_counts(data['label']), labels = [0,1], colors = ['blue', 'red'])
plt.axis('equal')
plt.show()


# In[6]:


import re
import nltk
from nltk.corpus import stopwords

def clean_sentence(sent):
    sentence = re.sub('<.*?>', ' ', sent) #removing html tags
    sentence = re.sub('[^\w\s]', ' ', sent) #removing punctutations
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sent) #removing single character
    sentence = re.sub(r'\s+', ' ', sent) #removing extra spaces
    sentence = re.sub('[^a-zA-Z]', ' ', sent) #removing numbers
    sentence = sentence.lower() #lower casing the words
    stop = stopwords.words('english') #introducing stopwords
    sentence = re.sub('[^A-Za-z0-9 ]+', ' ', sent) #removing special characters
    sentence = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', ' ', sent)
    sentence = ' '.join(text.lower() for text in sentence.split(' ')) #sometimes lowercasing doesn't work thus using this
    sentence = ' '.join(text for text in sentence.split() if text not in stop) #removing stopwords
    sentence = re.sub('\W+', ' ', sent)
    #sentence = ' '.join(text for text in sentence if text.isalnum())
    return sentence


# In[7]:


data['message'] = data['message'].apply(clean_sentence)


# In[8]:


data.head()


# In[9]:


from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

data['message'] = data['message'].apply(nltk.word_tokenize) #dividing strings into list of substrings


# In[10]:


data.head()


# In[11]:


stemmer = PorterStemmer() #keeps only th root word
data['message'] = data['message'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[12]:


data.head()


# In[13]:


data['message'] = data['message'].apply(lambda x: ' '.join(x)) #removing words from the list
data.head()


# In[14]:


count_vector = CountVectorizer()
vectorised = count_vector.fit_transform(data['message']) #convert words to sparse matrix for the model to understand


# In[16]:


transform = TfidfTransformer()
vectorised = transform.fit_transform(vectorised) #converting into based on frequency (tf) or normalising 
#to reduce the effect of frequency (tf-idf) on the model.


# In[17]:


#splitting the dataset into 80:20 ratio and shuffling them for random distribution.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectorised, data['label'], test_size= 0.2, shuffle=True, 
                                                    random_state= 40)


# In[23]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

dt = DecisionTreeClassifier()

params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(1,10),
          'min_samples_split': range(1,10),
          'min_samples_leaf': range(1,5)}

grid = GridSearchCV(dt, param_grid= params, cv = 10, verbose= 1, n_jobs= -1, scoring= 'roc_auc')

grid.fit(X_train, y_train)


# In[24]:


grid.best_score_


# In[26]:


grid.best_estimator_


# In[28]:


dt_grid = DecisionTreeClassifier(max_depth=9, min_samples_leaf=4, min_samples_split=9).fit(X_train, y_train)

dt_pred = dt_grid.predict(X_test)

from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(dt_pred, y_test))
print(roc_auc_score(y_test, dt_grid.predict_proba(X_test)[:,1]))


# In[36]:


from sklearn.metrics import confusion_matrix

print(sns.heatmap(confusion_matrix(dt_pred, y_test), annot = True , cmap= 'Greens'))


# In[32]:


from sklearn.metrics import roc_curve

fp, tp, threshold = roc_curve(y_test, dt_grid.predict_proba(X_test)[:,1])

plt.plot(fp, tp)
plt.xlabel("False Positives")
plt.ylabel("True Positives")
plt.show()


# Since the Area Under Curve is between 0.5 and 1 we can say that there is a high chance of our model to classify between positive and negative classes.

# In[ ]:




