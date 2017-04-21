
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import pandas as pd
from pandas.tools import plotting


# In[3]:

import urllib.request


# In[4]:

url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/sake_dataJ.txt'
urllib.request.urlretrieve(url, 'alcohol.txt')


# In[5]:

df = pd.read_csv('alcohol.txt', sep='\t') 


# In[6]:

df


# In[7]:

plotting.scatter_matrix(df.dropna(axis=1)[df.columns[:]], figsize=(10, 10)) 
plt.show()


# 突出して高いところがあることがわかる。  
# 相関係数は高そうに見える。

# In[8]:

pd.DataFrame(np.corrcoef(df.dropna().iloc[:, 1:].T.as_matrix().tolist()), 
             columns=df.columns[1:], index=df.columns[1:])


# In[9]:

print(pd.DataFrame(np.corrcoef(df.dropna().iloc[:, 1:].T.as_matrix().tolist()), 
             columns=df.columns[1:], index=df.columns[1:]))


# In[9]:

corrcoef = np.corrcoef(df.dropna().iloc[:, 1:].T.as_matrix().tolist())
plt.figure(figsize=(6, 5))
plt.imshow(corrcoef, interpolation='nearest', cmap=plt.cm.RdBu, vmin = -1, vmax = 1)
plt.colorbar()
tick_marks = np.arange(len(corrcoef))
plt.xticks(tick_marks, df.columns[1:], rotation=90)
plt.yticks(tick_marks, df.columns[1:])
plt.tight_layout()


# ほとんど真っ青で全ての相関係数が非常に高いことがわかる。
# つまりどの県でも酒を飲む人はこれら全ての種類の酒を飲む。

# 続いてPCAによる分析

# In[10]:

import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器


# In[11]:

dfs = df.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[12]:

dfs.index = df.iloc[:, 0]


# In[13]:

dfs


# In[14]:

#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :])
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :])
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(8, 8))
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.index):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.grid()
plt.show()


# 首都圏と大阪福岡などの大都市が塊から外れていることがわかる。

# In[15]:

plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# 第一成分と第二成分がほぼ全ての割合を占めていることがわかる。つまり都道府県別に見たときに都道府県の特徴はだいたい2種類あるということになる。  
# 第一成分は大都市が比較的外れており、東京が大きく外れていることから酒類の総消費量であると予測できる。(E=A+B+C+Dみたいな軸をとっている)  
# 第二成分は鹿児島大阪が外れており、東京がほぼゼロに近いことから、酒類の総消費量に対する日本酒の割合であることが予測できる。

# In[16]:

# 階層的クラスタリング
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs.iloc[:, :], metric = 'correlation',method = 'average')
plt.figure(figsize=(8, 8))
dendrogram(result1, orientation='right', labels=list(dfs.index), color_threshold=0.8)
plt.title("Dendrogram")
plt.xlabel("Threshold")
plt.grid()
plt.show()


# だいたい3つのグループに分けることができることがわかった

# In[17]:

pca = PCA()
pca.fit(dfs.iloc[:, :].T)
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :].T)
# 第一主成分と第二主成分でプロットする
plt.figure()
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.columns):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.grid()
plt.show()


# In[18]:

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# 第三成分までが大半の割合を占めていることがわかる。つまり酒を分類する成分は約3種類あるということがわかる

# In[19]:

# 第一成分と第三成分でプロット
plt.figure()
for x, y, name in zip(feature[:, 0], feature[:, 2], dfs.columns):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 2], alpha=0.8)
plt.grid()
plt.show()


# In[20]:

# 第二成分と第三成分でプロット
plt.figure()
for x, y, name in zip(feature[:, 1], feature[:, 2], dfs.columns):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 1], feature[:, 2], alpha=0.8)
plt.grid()
plt.show()


# 第一成分は焼酎、日本酒が高く、第二成分はワインが高い、第三成分は日本酒とビールが高い。

# In[21]:

# 階層的クラスタリング
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs.T.iloc[:, :], metric = 'correlation',method = 'average')
plt.figure(figsize=(8, 8))
dendrogram(result1, orientation='right', labels=list(dfs.columns), color_threshold=0.1)
plt.title("Dendrogram")
plt.xlabel("Threshold")
plt.grid()
plt.show()


# 焼酎と残りの酒の差が大きいことがわかった。

# In[ ]:



