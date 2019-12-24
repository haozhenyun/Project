# Homework1 :  Clustering with sklearn
##  


| **学号**  | 201914663  |   **姓名**  | 郝振云  |   **班级**  | 2019级学硕  |
## 
## 实验要求
	
测试以下聚类算法在上述两个数据集上的聚类效果
+ #### Datatsets
    * sklearn.datasets.load_di gits
    * sklearn.datasets.fetch_2 0newsgroups

	
+ #### Algorithm
    * K-Means
    * AffinityPropagation
    * MeanShift
    * SpectralClustering
    * WardhierarchialClustering
    * AgglomerativeClustering
    * DBSCAN
    * GaussianMixture
+ #### Evaluation
    *  labels_true and labels_pred 
    *  Normalized Mutual Information (NMI) 
    *  Homogeneity: each cluster contains only members of a single class 
    *  Completeness: all members of a given class are assigned to the same cluster 


## 
## 实验过程

实验的数据集有两个，下面是对两个数据集的处理
###### 数据集1  sklearn.datasets.load_di gits
```python
# 加载数据集 sklearn.datasets.load_di gits，
# 该数据集无需进行特殊处理，直接使用scale进行处理即可
from sklearn.datasets import load_digits
digits = load_digits()
# 对数据集进行归一化缩放处理
data = scale(digits.data)
```
###### 数据集2  sklearn.datasets.fetch_2 0newsgroups
```python
# 该数据集为一个文本文档数据集，需要提前对数据进行处理
# 使用TfidfVectorizer来处理数据
vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english')
X = vectorizer.fit_transform(dataset.data)
# X.shape = (3387, 10000)
print()
# 使用LSA降维
svd = TruncatedSVD(64)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)
```

使用sklearn中的算法对处理后的数据进行处理，以K-Means为例
```python
# 调用K-Means方法进行处理
km = KMeans(n_clusters=10)
```
后续使用实验要求中的评估策略对结果进行评估并输出
```python
# 进行多种评估
estimator.fit(data)
#print("labels_true")
print(labels)
#print("labels_pred")
print(estimator.labels_)
print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f'
	% (name, (time() - t0),
		metrics.homogeneity_score(labels, estimator.labels_),
		metrics.completeness_score(labels, estimator.labels_),
		metrics.normalized_mutual_info_score(labels, estimator.labels_)))
```

## 
## 实验结果及分析  
	
	
分类结果  
	
	数据集1 
![sklearn.datasets.load_di gits](Figure_1.png) 
 
	数据集2 
![sklearn.datasets.fetch_2 0newsgroups](Figure_2.png) 
	
	
	
评估结果 
	
数据集1
```python
n_digits: 10, 	 n_samples 1797, 	 n_features 64
__________________________________________________________________________________
init		time	homo	compl	NMI	labels			labels_pred
KMeans   	0.19s	0.602	0.650	0.626	[0 1 2 ... 8 9 8]	[0 3 3 ... 3 7 7]
APropagation	6.03s	0.932	0.460	0.655	[0 1 2 ... 8 9 8]	[102  86   3 ... 100  34   2]
MeanShift	5.50s	0.014	0.281	0.063	[0 1 2 ... 8 9 8]	[0 0 0 ... 0 0 0]
SpClustering	428.34s	0.001	0.271	0.012	[0 1 2 ... 8 9 8]	[2 2 2 ... 2 2 2]
WcClustering	0.17s	0.758	0.836	0.797	[0 1 2 ... 8 9 8]	[5 1 1 ... 1 1 1]
AgClustering	0.13s	0.017	0.249	0.065	[0 1 2 ... 8 9 8]	[0 0 0 ... 0 0 0]
DBSCAN   	0.33s	0.000	1.000	0.375	[0 1 2 ... 8 9 8]	[-1 -1 -1 ... -1 -1 -1]
GauMixture	2.28s	0.493	0.588	0.539	[0 1 2 ... 8 9 8]	[7 2 3 ... 2 2 4]
```
	
数据集2
```python
__________________________________________________________________________________
init		time	homo	compl	NMI 		labels 		labels_pred
KMeans   	0.03s	0.510	0.512	0.511	[0 1 1 ... 2 1 1]	[1 2 2 ... 3 2 2]
APropagation	11.99s	0.768	0.202	0.394	[0 1 1 ... 2 1 1]	[ 71  60 161 ... 142 137 116]
MeanShift	15.81s	0.000	1.000	0.000	[0 1 1 ... 2 1 1]	[0 0 0 ... 0 0 0]
SpClustering	1.58s	0.478	0.477	0.478	[0 1 1 ... 2 1 1]	[3 1 1 ... 0 1 1]
WcClustering	0.55s	0.277	0.367	0.319	[0 1 1 ... 2 1 1]	[0 1 1 ... 0 1 1]
AgClustering	0.50s	0.026	0.031	0.028	[0 1 1 ... 2 1 1]	[1 1 2 ... 0 1 2]
DBSCAN   	1.53s	0.427	0.218	0.305	[0 1 1 ... 2 1 1]	[-1 -1 -1 ... 72 -1 -1]
GauMixture	0.48s	0.593	0.594	0.593	[0 1 1 ... 2 1 1]	[2 1 1 ... 0 1 1]
```
