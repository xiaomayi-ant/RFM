import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

#设置图表显示格式
matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

#数据清洗
df=pd.read_excel(r'C:\Users\ant.zheng\Desktop\blog-0\销售额.xls')
#提出重复值
df.drop_duplicates(subset='需求ID',keep='first',inplace=True)
df=df[(pd.to_datetime(df['需求提交时间'])>=pd.to_datetime('20201001'))&(pd.to_datetime(df['需求提交时间'])<=pd.to_datetime('20201101'))]
print(df['需求提交时间'])
#素材币数据分布
plt.figure(figsize=(10,5))
sns.distplot(df['合计素材币'])
plt.show()
#处理R值
df['需求提交时间']=pd.to_datetime(df['需求提交时间'])
df['Datediff']=(pd.to_datetime('today')-df['需求提交时间']).dt.days

#数据分析
R_Agg=df.groupby('需求人')['Datediff'].agg([('距今最近一次消费','min')])         #groupby和agg函数的合并使用在于汇总，按行，求值按列
F_Agg=df.groupby('需求人')['需求数量'].agg([('2020年10月份消费频次','count')])   #按照需求数量合计计算
M_Agg=df.groupby('需求人')['合计素材币'].agg([('2020年10月份消费金额',sum)])     #
rfm=R_Agg.join(F_Agg).join(M_Agg)    #join函数按行合并，对列进行合并（列表，元组）
print(rfm)

#K-means聚类
from  sklearn import  preprocessing
import matplotlib
import matplotlib.pyplot  as  plt
from  sklearn.cluster import KMeans

#数据标准化
min_max_scaler = preprocessing.MinMaxScaler()
rfm_norm = min_max_scaler.fit_transform(rfm)
#算法评测
# 肘部法则|SSE
distortions =[]
for i in range(1,10):
    km= KMeans(n_clusters=i,
               init='k-means++',
               n_init=10,
               max_iter=300,
               random_state=0)
    km.fit(rfm_norm)
    distortions.append(km.inertia_)
fig=plt.figure(figsize=(7,4))
plt.plot(range(1,10),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortions')

#轮廓系数评测法则
import time
from sklearn.metrics import silhouette_score,calinski_harabasz_score
clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
silhouette_scores = []
#轮廓系数silhouette_scores
print('start time: ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
for k in clusters:
    y_pred = KMeans(n_clusters = k, verbose = 0, n_jobs = -1, random_state=1).fit_predict(rfm_norm)
    score = silhouette_score(rfm_norm, y_pred)#silhouette_score：所有样本的轮廓系数的平均值，silhouette_sample：所有样本的轮廓系数
    silhouette_scores.append(score)
print('finish time: ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
fig1=plt.figure(figsize=(7,4))
plt.plot(clusters, silhouette_scores, '*-')
plt.xlabel('k值')
plt.ylabel('silhouette_score')
plt.show()
#利用肘部法则确定k值，利用轮廓系数评估聚类效果   可选定K=6
# 聚类
plt.scatter(rfm_norm[:,1], rfm_norm[:,2], c="red", marker='*')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()
#构造聚类器
clf=estimator = KMeans(n_clusters=6,random_state=1)  # 构造聚类器
clf.fit(rfm_norm)  # 聚类
#原始数据聚类标签
rfm['label']=clf.labels_
result=rfm['label'].value_counts()
print("各个标签的数目\n",result)   #各个标签或类的数目
print("原始数据的分类结果:\n",rfm.head())


#可视化聚类结果，可分为两种情况（1,按照下述类别 2,通过概率密度图）
label_pred = clf.labels_  # 获取聚类标签
x0 = rfm_norm[label_pred == 0]
x1 = rfm_norm[label_pred == 1]
x2 = rfm_norm[label_pred == 2]
x3 = rfm_norm[label_pred == 3]
x4 = rfm_norm[label_pred == 4]
x5 = rfm_norm[label_pred == 5]

#分离数据，可视化
from mpl_toolkits.mplot3d import Axes3D
x0x=x0[:, 0]
x0y=x0[:, 1]
x0z=x0[:, 2]
x1x=x1[:, 0]
x1y=x1[:, 1]
x1z=x1[:, 2]
x2x=x2[:, 0]
x2y=x2[:, 1]
x2z=x2[:, 2]
x3x=x3[:, 0]
x3y=x3[:, 1]
x3z=x3[:, 2]
x4x=x4[:, 0]
x4y=x4[:, 1]
x4z=x4[:, 2]
x5x=x5[:, 0]
x5y=x5[:, 1]
x5z=x5[:, 2]

# plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
# plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
# plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
# plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='+', label='label3')
# plt.scatter(x4[:, 0], x4[:, 1], c="gray", marker='o', label='label4')
# plt.scatter(x5[:, 0], x5[:, 1], c="cyan", marker='*', label='label5')

# 绘制3D散点图
fig2 = plt.figure()
ax = Axes3D(fig2)
ax.scatter(x0x, x0y, x0z, c='r', label='类1')
ax.scatter(x1x, x1y, x1z, c='g', label='类2')
ax.scatter(x2x, x2y, x2z, c='b', label='类3')
ax.scatter(x3x, x3y, x3z, c='gray', label='类4')
ax.scatter(x4x, x4y, x4z, c='cyan', label='类5')
ax.scatter(x5x, x5y, x5z, c='yellow', label='类6')

#绘制图例
ax.legend(loc='best')

#添加坐标轴(顺序是z,y,x)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()
rfm.to_excel(r'C:\Users\ant.zheng\Desktop\素材币10月份聚类结果.xlsx')
print("写入成功")

#绘制聚类后的概率密度图
import matplotlib.pyplot as plt
k=6
n=3
l=0
plt.figure(figsize=(2.5* n, 6.5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示正负号
for i  in  range(k):
    data=rfm[rfm['label']==i]
    for j  in  range(3):
        l += 1
        if l>=4:
            l=1
        plt.subplot(n, 1, l)
        p=data.iloc[:, j].plot(kind='kde', linewidth=2, subplots=True, sharex=False)
        plt.legend()
    plt.suptitle(str(i)+'类别客户概率密度图')
    plt.savefig(r'C:\Users\ant.zheng\Desktop\%s.jpg' %(str(i)+'类别'))
    plt.show()






