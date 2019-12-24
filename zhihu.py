#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# In[6]:



with open('/home/administrator/share/hzy/data/data.pkl','rb') as file:
    data_a=pickle.load(file)
data_a.shape


# In[7]:


with open('/home/administrator/share/hzy/data/member_feat.pkl','rb') as file:
    member_info=pickle.load(file)
member_info.shape


# In[8]:


pd.options.display.max_columns=None  #显示dataframe所有列数
member_info.head()


# In[11]:


with open('/home/administrator/share/hzy/data/ques_feat.pkl','rb') as file:
    question_info=pickle.load(file)
question_info.shape


# In[12]:


question_info.head()


# In[13]:


data_a.head()


# In[17]:


columns=['uid']
for i in range(64):
    columns.append('follow_topic_{}'.format(i))
member_topic=member_info[columns]


# In[18]:


member_topic.head()


# In[19]:


data_a=pd.merge(data_a,member_topic,how='left',left_on='uid',right_on='uid')


# In[20]:


columns=['qid']
for i in range(64):
    columns.append('topic_vector_{}'.format(i))
question_topic=question_info[columns]

question_topic.head()


# In[22]:



data_a.head()


# In[23]:


data_a.shape


# In[24]:


data_a=pd.merge(data_a,question_topic,how='left',left_on='qid',right_on='qid')
data_a.shape


# In[32]:


with open('/home/administrator/share/hzy/data/data_a.pkl','wb') as file:
    pickle.dump(data_a,file)


# In[26]:


train_label=2593669
feature_cols = [x for x in data_a.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]
# target编码
#print("feature size %s", len(feature_cols))

X_train_all = data_a.iloc[:train_label][feature_cols]
y_train_all = data_a.iloc[:train_label]['label']
test = data_a.iloc[train_label:]

print(X_train_all.shape, y_train_all.shape, test.shape)


# In[ ]:


fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#model_lgb = LGBMClassifier(n_estimators=2000, n_jobs=-1, objective='binary', seed=1000, silent=True)
model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.01, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=-1, silent=True)


for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols],                                      y_train_all.iloc[train_idx],                                      y_train_all.iloc[val_idx]
    model_lgb.fit(X_train, y_train,
                  eval_metric=['logloss', 'auc'],
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=10)
sub = pd.read_csv('/home/administrator/share/hzy/data/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
sub.columns = ['qid', 'uid', 'dt']
sub['label'] = model_lgb.predict_proba(test[feature_cols])[:, 1]
sub.to_csv('/home/administrator/share/hzy/data/result.txt', index=None, header=None, sep='\t')


# In[ ]:




