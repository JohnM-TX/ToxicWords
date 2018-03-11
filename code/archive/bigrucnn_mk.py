
# coding: utf-8

# In[1]:


import time
start_time = time.time()

import sys, os, re, csv, codecs
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
# from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.layers import GRU, BatchNormalization
from keras.models import Model, load_model
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.callbacks import Callback

import markovify as mk



np.random.seed(32)
os.environ['OMP_NUM_THREADS'] = "4"

train0 = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


class_names = list(train0)[-6:]


train0['comment_text'].fillna("no comment")
test['comment_text'].fillna("no comment")

multarray = np.array([100000, 10000, 1000, 100, 10, 1])
train0['true_class'] = np.sum(train0.iloc[:, 2:8].values * multarray, axis=1)
train0[5:7]

aug1 = train0[train0['true_class'] == 101110] 
count1 = 2500

aug2 = train0[train0['true_class'] == 111011]
count2 = 3500

aug_list = aug1['comment_text'].tolist() 
nchar = int(aug1['comment_text'].str.len().median())

mkv_text = []
text_model = mk.Text(aug_list)
for i in range(count1):
        new = text_model.make_short_sentence(nchar)
        mkv_text.append(new)

aug_text = pd.Series(mkv_text, name='comment_text')    

ys = np.array([[1,0,1,1,1,0],]*count1)

aug_ys = pd.DataFrame(ys, columns=class_names)    
aug_ys.head()
    
augdf = aug_ys.join(aug_text)
#     return augdf
    
# train_class = np.ones(count)
# train_target = np.append(train_base_tgt, train_class)
    

aug1df = augdf
    

    
    
aug_list = aug2['comment_text'].tolist() 
nchar = int(aug2['comment_text'].str.len().median())

mkv_text = []
text_model = mk.Text(aug_list)
for i in range(count2):
        new = text_model.make_short_sentence(nchar)
        mkv_text.append(new)

aug_text = pd.Series(mkv_text, name='comment_text')    

ys = np.array([[1,1,1,0,1,1],]*count2)

aug_ys = pd.DataFrame(ys, columns=class_names)    
aug_ys.head()
    
augdf = aug_ys.join(aug_text)

aug2df = augdf


train0.drop(columns=['id', 'true_class'], inplace=True)
train = pd.concat([train0, aug1df, aug2df], axis=0, join='outer', ignore_index=True,)



y = train[class_names].values

X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))



# embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
embedding_path = "../vecs/glove.840B.300d.txt"
# embedding_path = "../vecs/numberbatch.txt"

embed_size = 300
max_features = 100000 # 100000
max_len = 150 # 100, 150



# In[5]:


raw_text_train = X_train["comment_text"].str.lower()
raw_text_valid = X_valid["comment_text"].str.lower()
raw_text_test = test["comment_text"].str.lower()

tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text_train)
X_train["comment_seq"] = tk.texts_to_sequences(raw_text_train)
X_valid["comment_seq"] = tk.texts_to_sequences(raw_text_valid)
test["comment_seq"] = tk.texts_to_sequences(raw_text_test)

X_train = pad_sequences(X_train.comment_seq, maxlen = max_len)
X_valid = pad_sequences(X_valid.comment_seq, maxlen = max_len)
test = pad_sequences(test.comment_seq, maxlen = max_len)


# In[6]:


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
print("Embedded index")


# In[ ]:


word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:




file_path = "./weights/best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval = 1)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)

batchsz = 128  # 128

def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (max_len,))
    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(units, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    x = Dense(6, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, Y_train, batch_size = batchsz, epochs = 5, validation_data = (X_valid, Y_valid), 
                        verbose = 1, callbacks = [ra_val, check_point, early_stop])
    model = load_model(file_path)
    return model

    


# In[ ]:


model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)
pred = model.predict(test, batch_size = 1024, verbose = 2)
pred = np.around(pred, decimals=8)

# In[ ]:


submission = pd.read_csv("../input/sample_submission.csv")
submission[class_names] = (pred)
submission.to_csv("../subs/sub_bigrucnn_mk_5eps.csv", index = False)
print("[{}] Completed!".format(time.time() - start_time))


# In[ ]:


submission.head()
