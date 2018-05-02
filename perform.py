
# coding: utf-8

# In[2]:

import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:

iris.data


# In[4]:

len(iris.data)


# In[5]:

iris.target


# In[6]:

len(iris.target)


# In[7]:

# モデルを作成
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)


# In[8]:

predicted = clf.predict(iris.data)


# In[9]:

sum(predicted == iris.target) / len(iris.target)


# In[10]:

tree.export_graphviz(clf, out_file="tree.dot",
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True)


# In[ ]:



