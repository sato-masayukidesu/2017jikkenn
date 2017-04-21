
# coding: utf-8

# In[15]:

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from pandas.tools import plotting
import urllib.request
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
from scipy.cluster.hierarchy import linkage, dendrogram


# In[4]:

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
urllib.request.urlretrieve(url, 'awabi.txt')


# In[5]:

df = pd.read_csv('awabi.txt', sep=',', header=None)


# In[6]:

df.columns = ["Sex", 
              "Length",
              "Diameter",
              "Height",
              "Whole_weight",
              "Shucked_weight",
              "Viscera_weight",
              "Shell_weight",
              "Rings"]


# In[7]:

df


# In[8]:

color_codes = ["#FF0000", "#0000FF", "#00FF00"]
class_names = list(set(df.iloc[:, 0]))
colors = [color_codes[class_names.index(x)] for x in list(df.iloc[:, 0])]
plotting.scatter_matrix(df.dropna(axis=1)[df.columns[:]], figsize=(20, 20), color=colors) 
plt.show()


# Rings以外が正の相関を示していることがわかる。  
# Ringsについてはノイズが入っているように見えるが、ここは整数しか入っていないためこのような散布図になる。  
# Iの方が年齢が低く、MFの方が全体的に高い値を示している。

# In[10]:

pd.DataFrame(np.corrcoef(df.dropna().iloc[:, 1:].T.as_matrix().tolist()), 
             columns=df.columns[1:], index=df.columns[1:])


# In[11]:

corrcoef = np.corrcoef(df.dropna().iloc[:, 1:].T.as_matrix().tolist())
plt.figure(figsize=(6, 5))
plt.imshow(corrcoef, interpolation='nearest', cmap=plt.cm.RdBu, vmin = -1, vmax = 1)
plt.colorbar()
tick_marks = np.arange(len(corrcoef))
plt.xticks(tick_marks, df.columns[1:], rotation=90)
plt.yticks(tick_marks, df.columns[1:])
plt.tight_layout()


# 相関係数はRingsを除いて全て高く、正である。また、Heightのみ少し低い。  
# つまり、アワビの重さや大きさは全て関連しており、それらは年齢にはあまり寄らないということがわかる。
# 
# 続いてPCAによる分析

# In[12]:

dfs = df.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[13]:

dfs


# In[17]:

#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :])
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :])
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# 第三成分まででおおよそ表せていることがわかる。

# In[26]:

color_codes = ["#FF0000", "#0000FF", "#00FF00"]
class_names = list(set(df.iloc[:, 0]))
colors = [color_codes[class_names.index(x)] for x in list(df.iloc[:, 0])]
#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :])
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :])
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(8, 8))
for x, y, name in zip(feature[:, 0], feature[:, 1], ""):
    plt.text(x, y, name, alpha=0.5, size=10)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, color=colors)
plt.grid()
plt.show()


# In[32]:

# 第一主成分と第三主成分でプロットする
plt.figure(figsize=(8, 8))
for x, y, name in zip(feature[:, 0], feature[:, 2], ""):
    plt.text(x, y, name, alpha=0.5, size=10)
plt.scatter(feature[:, 0], feature[:, 2], alpha=0.8, color=colors)
plt.grid()
plt.show()


# In[33]:

# 第二主成分と第三主成分でプロットする
plt.figure(figsize=(8, 8))
for x, y, name in zip(feature[:, 1], feature[:, 2], ""):
    plt.text(x, y, name, alpha=0.5, size=10)
plt.scatter(feature[:, 1], feature[:, 2], alpha=0.8, color=colors)
plt.grid()
plt.show()


# 全てにおいてM,F,Iをうまく分離できていないことがわかる。

# 今回のデータセットでは、アワビのオスメス子供はうまく分離できないことがわかった。  
# 年齢

# In[ ]:



