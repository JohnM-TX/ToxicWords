
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('autosave', '600')


# In[9]:


import numpy as np
import pandas as pd

from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import os 
from glob import glob

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

classes = train.columns[-6:]
print(classes)


# In[10]:



files = glob(os.path.join(os.pardir, 'ensembles', 'pred*.csv'))
# neworder=[2,0,4,1,5,3]
# files = [files[n] for n in neworder]
files


# In[ ]:


file_list = []
for i, f in enumerate(files):
    file_df = pd.read_csv(f)
    if file_df.columns[0] == 'id':
        file_df.drop(['id'], axis=1, inplace=True)
    file_list.append(file_df)


# In[ ]:


preds = np.zeros_like(p_list[0])
for i, f in enumerate(files):
    model

