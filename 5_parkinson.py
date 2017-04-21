
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd

from pandas.tools import plotting
import urllib.request
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
from sklearn import cross_validation as cv


# In[ ]:

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
urllib.request.urlretrieve(url, 'pa-kinson.txt')


# In[2]:

df = pd.read_csv('pa-kinson.txt', sep=',')


# In[3]:

df


# In[4]:

df.index = list(range(1, len(df.index)+1))


# In[5]:

color_codes = ["#FF0000", "#0000FF", "#00FF00"]
class_names = list(set(df.iloc[:, -7]))
colors = [color_codes[class_names.index(x)] for x in list(df.iloc[:, -7])]
plotting.scatter_matrix(df[list(df.columns[:])], figsize=(30, 30), color=colors) 
plt.show()


# 赤が健康体、青がパーキンソン病の人であり、一部の散布図では分離できそうなことが読み取れる。

# In[6]:

# 名前と評価値を抜くためのブーリアンを作成
libool = [True] * len(df.columns)
libool[-7] = False
libool[0] = False
# 行列の正規化
dfs = df.iloc[:, libool].apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[7]:

color_codes = ["#FF0000", "#0000FF", "#00FF00"]
class_names = list(set(df.iloc[:, -7]))
colors = [color_codes[class_names.index(x)] for x in list(df.iloc[:, -7])]
#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :])
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :])
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(8, 8))
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.index):
    plt.text(x, y, "", alpha=0.5, size=10)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=colors)
plt.grid()
plt.show()


# 少し分離できているように思う。

# In[8]:

plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# 寄与率の比較的高い成分が半分ほどになったため、主成分分析はある程度有効であると考える

# In[16]:

feature_names = df.columns[:-7]
target_names = list(set(df.iloc[:, -7]))
sample_names = df.index
data = df.iloc[:, libool]
target = df.iloc[:, -7]


# In[17]:

data


# In[11]:

target


# In[36]:

train_data, test_data, train_target, test_target = cv.train_test_split(data, target, test_size=0.5)


# In[37]:

train_data


# In[38]:

# 様々なパラメータ（ハイパーパラメータという）で学習し、分離性能の最も良いモデルを選択する。
parameters = [
    {'kernel': ['linear'], 'C': [1]},
    {'kernel': ['rbf'],     'C': [1], 'gamma': [1e-2]},      
    {'kernel': ['poly'],'C': [1], 'degree': [2]}]


# In[39]:

from sklearn import svm
from sklearn.metrics import accuracy_score
import time
start = time.time()
from sklearn import grid_search

# train_data を使って、SVM による学習を行う
gs = grid_search.GridSearchCV(svm.SVC(), parameters, n_jobs=-1).fit(train_data, train_target)
# 分離性能の最も良かったモデルが何だったか出力する
print(gs.best_estimator_)

# モデル構築に使わなかったデータを用いて、予測性能を評価する
pred_target = gs.predict(test_data)
print ("Accuracy_score:{0}".format(accuracy_score(test_target, pred_target)))
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))


# 精度は85%ほどで時間は10秒以内である。  
# 

# In[40]:

df = pd.DataFrame(columns=['test', 'pred'])
df['test'] = test_target
df['pred'] = pred_target
df.T


# In[41]:

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_target, pred_target)
pd.DataFrame(cm)


# In[42]:

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.coolwarm)
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')


# これを見るとパーキンソン病の人はきちんと判別できているが、健康な人に対しての誤診が多く、半分ほどいるとわかる。

# In[ ]:



