#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv("affaits_dataset.csv",header=0)
data.head()


# In[13]:


le = preprocessing.LabelEncoder()

data = data.apply(le.fit_transform)
data.head()


# In[14]:


y = data["affairs"]
x = data[list(data.columns[2:12])]


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[16]:


Tree = tree.DecisionTreeClassifier()
Tree = Tree.fit(x_train, y_train)

y_pred = Tree.predict(x_test)


# In[17]:


print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


# In[18]:


features = pd.DataFrame()
features['feature'] = data.columns[2:12]
features['importance'] = Tree.feature_importances_
features.sort_values(by=['importance'])
features


# In[19]:


Forest = RandomForestClassifier(n_estimators = 100)
Forest = Forest.fit(x_train, y_train)

y_pred = Forest.predict(x_test)


# In[20]:


print("confusion matrix:", confusion_matrix(y_test, y_pred), sep="\n")
print("accuracy score:", accuracy_score(y_test,y_pred), sep="\n")


# In[21]:


features = pd.DataFrame()
features['feature'] = data.columns[2:12]
features['importance'] = Forest.feature_importances_
features.sort_values(by=['importance'])
features


# In[22]:


# need to install first
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import graphviz 

dot_data = tree.export_graphviz(Tree, out_file=None, 
                      feature_names=data.columns[2:12],  
                      class_names=data.columns[1],  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# In[ ]:




