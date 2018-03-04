

import os
import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[3]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback


# In[4]:


import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '4'


# In[5]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')

X_train = train["comment_text"].fillna("fillna").values
y_train = list(train)[-6:]
X_test = test["comment_text"].fillna("fillna").values


max_features = 30000
maxlen = 100   
embed_size = 300


# In[6]:



#preprocess
# X_train = [x.replace('!', " ! ") for x in X_train]
# X_train = [x.replace('?', " ! ") for x in X_train]


tokenizer = text.Tokenizer(num_words=max_features,
                          filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
#                          oov_token='misspelled')

tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


# In[7]:


# EMBEDDING_FILE = '../input/wiki.en.vec'
EMBEDDING_FILE = '../input/numberbatch-en.txt'

def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[8]:


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


# In[9]:


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()


# In[10]:


model.summary()


# In[ ]:


batch_size = 32
epochs = 3 # 6

classes = list(train)[-6:]
y_train = train[classes].values

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)


# In[ ]:


hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)
# 32% GPU


# In[ ]:


y_pred = model.predict(x_test, batch_size=1024)


# In[ ]:


submission[classes] = np.around(y_pred, decimals=4)
submission.to_csv('../subs/sub_GRU2.csv', index=False)


# In[ ]:


7

