#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from itertools import combinations


# In[3]:


#### predict
read_file = pd.read_csv ('test-public.txt')

df_pre = pd.read_csv(
    'test-public.txt', sep="\t")
df_pre.columns = ['Id','node1', 'node2']
#data_pre = df_pre.drop(columns=['Id'])
df_pre.head(1)


# In[4]:


df_pre_list = df_pre.values.tolist()
print(df_pre_list)


# The training data is too big, which makes is difficult to build a model.
# Therefore, we are only going to use the followers whose number of followees are less than 1000.

# In[5]:


ab = []

for x in df_pre_list:
    ab.append(str(x[1]))
    ab.append(str(x[2]))

# remove duplicate items from the list
ab = list(dict.fromkeys(ab))
print(random.sample(ab,1), "len: ",len(ab))


# In[6]:


file = 'train_part1.txt'
with open(file) as file:
    rows = (line.split('\t') for line in file)
    data_list_part1 = [row[0:] for row in rows]
for x in data_list_part1:
    x[-1] = x[-1].rstrip('\n')
len(data_list_part1)


# In[7]:


file = 'train.txt'
with open(file) as file:
    rows = (line.split('\t') for line in file)
    data_list = [row[0:] for row in rows]
for x in data_list:
    x[-1] = x[-1].rstrip('\n')
len(data_list)


# In[57]:


# 1st iteration
abab = []
x_list = []
for x in ab:
    for y in data_list:
        if (100 < len(y) < 200) and (x in y):
            abab.append(y)
            x_list.append(x)
            break
            
print("1st: ",len(x_list))

# 2nd iteration
ab2 = []
for i in ab:
    if i not in x_list:
        ab2.append(i)

for x in ab2:
    for y in data_list:
        if (200 < len(y) < 300) and (x in y):
            abab.append(y)
            x_list.append(x)
            break
            
print("2nd: ",len(x_list))

# 3rd iteration
ab3 = []
for i in ab2:
    if i not in x_list:
        ab3.append(i)

for x in ab3:
    for y in data_list:
        if (3000 < len(y) < 400) and (x in y):
            abab.append(y)
            x_list.append(x)
            break
            
print("3rd: ",len(x_list))


# In[45]:


#print(abab)


# In[61]:


short_data_list = []
for x in data_list_part1:
    if 2 < len(x) < 1000 :
        short_data_list.append(x)
print(len(short_data_list))


# In[62]:


print(len(abab))
print(len(short_data_list))


# Also, import the dataset to predict the probability at the end.

# In[12]:


# get samples----------- @dataset_sample
#sample_count = 20
#data_list_sample = random.sample(short_data_list, sample_count)


# In[13]:


# dic
#data_dict = {x[0]:x[1:] for x in data_list_sample}


# In[63]:


short_data_list = short_data_list + abab
#print(short_data_list)


# In[64]:


# dic - removes duplicates
data_dict = {x[0]:x[1:] for x in short_data_list}


# In[65]:


pairs = []
for key in data_dict: #each row
    for val in data_dict[key]:
        pair = [key,val]
        pairs.append(pair)

node1_list = []
node2_list = []
for x in pairs:
    node1_list.append(x[0])
    node2_list.append(x[1])

df = pd.DataFrame({'node1': node1_list, 'node2': node2_list})
df.shape
df.head()


# In[66]:


G = nx.from_pandas_edgelist(df, "node1", "node2",create_using=nx.Graph() )


# In[20]:


# plot graph
#plt.figure(figsize=(10,10))

#pos = nx.random_layout(G, seed=23)
#nx.draw(G, with_labels=False,  pos = pos, node_size = 4, alpha = 0.6, width = 0.1)

#plt.show()


# In[67]:


# combine all nodes in a list
node_list = node1_list + node2_list
# remove duplicate items from the list
node_list = list(dict.fromkeys(node_list))
len(node_list)


# In[82]:



#print(abab)

node_list_long = []
for x in abab:
    for y in x:
        node_list_long.append(y)
node_list_unique = list(dict.fromkeys(node_list_long))

node_list_short = list(random.sample(node_list_unique,4000))
node_list_short = node_list_short + ab
print(len(node_list_short))
node_list_short = list(dict.fromkeys(node_list_short))
node_list = node_list_short


# In[85]:


df_list = df.values.tolist()
print(df_list)


# In[83]:


pairs_all = []
for x in tqdm(range(len(node_list))):
    for y in range(x+1,len(node_list)):
        if nx.has_path(G,node_list[x], node_list[y]) == True:
            if nx.shortest_path_length(G,node_list[x],node_list[y]) <=2:
                pairs_all.append([node_list[x], node_list[y]])


# In[23]:


no_edge_list = []
for x in pairs_all:
    if x not in pairs:
        no_edge_list.append(x)
        
no_edge_node1 = []
no_edge_node2 = []
for x in no_edge_list:
    no_edge_node1.append(x[0])
    no_edge_node2.append(x[1])
         
data = pd.DataFrame({'node1':no_edge_node1, 'node2':no_edge_node2})

data['link'] = 0


# In[24]:


print(len(no_edge_list),len(pairs))


# In[25]:


initial_node_count = len(G.nodes)
df_temp = df.copy()
removable_edges = []

for i in tqdm(df.index.values):
    #remove a node pair and build a new graph
    G_temp = nx.from_pandas_edgelist(df_temp.drop(index=i), "node1", "node2", create_using=nx.Graph())
    # check if graph is still valid
    if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
        removable_edges.append(i)
        df_temp = df_temp.drop(index=i)

        
print(len(removable_edges))


# In[ ]:


# create dataframe of removable edges
df_ghost = df.loc[removable_edges]

# add the target variable 'link'
df_ghost['link'] = 1

data = data.append(df_ghost[['node1', 'node2', 'link']], ignore_index=True)


# In[ ]:


data['link'].value_counts()


# In[ ]:


# drop removable edges
df_partial = df.drop(index=df_ghost.index.values)

# build graph
G_data = nx.from_pandas_edgelist(df_partial, "node1", "node2", create_using=nx.Graph())


# In[ ]:


from node2vec import Node2Vec

# Generate walks
node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

# train node2vec model
n2w_model = node2vec.fit(window=3, min_count=1)


# In[ ]:


x = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(data['node1'], data['node2'])]


# In[ ]:


data['link'].value_counts()


# In[ ]:


df_pre_list_e = []
for x in df_pre_list:
    if (str(x[1]) in node_list) and (str(x[2]) in node_list):
        print(x)
        dfdp_e.append(x)
        #dfdp.remove(x)
        
print(len(df_pre_list_e))
    


# In[ ]:


a = []
b = []
for x in df_pre_list_e:
    a.append(str(x[0])) ### check if above result is string
    b.append(str(x[1]))


# In[ ]:


XX = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(a,b)]


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.3, 
                                                random_state = 35)


# In[ ]:


lr = LogisticRegression(class_weight="balanced")

lr.fit(xtrain, ytrain)


# In[ ]:


predictions = lr.predict_proba(xtest)


# In[ ]:


predictions_pre = lr.predict_proba(XX)


# In[ ]:


predictions_pre


# In[ ]:


roc_auc_score(ytest, predictions[:,1])

