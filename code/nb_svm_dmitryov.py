
# coding: utf-8

# In[2]:


# %load nb_svm_dmitryov.py

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

lens = train.comment_text.str.len()


label_cols = train.columns[-6:]
train['none'] = 1-train[label_cols].max(axis=1)

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

print('vectorizing')
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])


def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


x = trn_term_doc
test_x = test_term_doc


# In[7]:


from sklearn.model_selection import cross_val_score, cross_val_predict

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    cv_score = np.mean(cross_val_score(m, x_nb, y, cv=3, scoring='roc_auc'))
    cv_preds = cross_val_predict(m, x_nb, y, cv=3, method='predict_proba')
    return m.fit(x_nb, y), r, cv_score, cv_preds



train_preds = pd.DataFrame.from_dict({'id': train['id']})
test_preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r,cv_score, cv_preds = get_mdl(train[j])
    print('CV score for class {} is {}'.format(j, cv_score))
    train_preds[j] = cv_preds[:, 1]
    test_preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


    
train_preds.to_csv('../ensembles/preds_nb_svm.csv', index=False)
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(test_preds, columns = label_cols)], axis=1)
submission.to_csv('../ensembles/test_nb_svm.csv', index=False)


