
# coding: utf-8

# In[ ]:


# %load bi_lstm_cv.py

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras import layers as lrs
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint


# get data
train = pd.read_csv("../input/train.csv", nrows=None)
test = pd.read_csv("../input/test.csv", nrows=None)


# set vars
max_features = 100000
embed_size = 300
max_length = 150         #150


# preprocess
print('preprocessing')

class_names = list(train)[-6:]
y_train = train[class_names].values

multarray = np.array([100000, 10000, 1000, 100, 10, 1])
y_multi = np.sum(train[class_names].values * multarray, axis=1)

train['comment_text'].fillna("no comment")
test['comment_text'].fillna("no comment")

x_train0 = train['comment_text']
x_test0 = test['comment_text']

toker = text.Tokenizer(num_words=max_features, 
    filters='?!"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n',
    lower=True)
toker.fit_on_texts(list(x_train0)+list(x_test0))

x_train = toker.texts_to_sequences(x_train0)
x_test = toker.texts_to_sequences(x_test0)

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)


# prepare word vectors/matrix
print('vectorizing')

embeddings_index = {}
embedding_file = "../vecs/crawl-300d-2M.vec" 
# embedding_file = "../vecs/glove.840B.300d.txt" 

with open(embedding_file, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs          
               
word_idx = toker.word_index

num_words = min(max_features, len(word_idx) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_idx.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector  # words not found will be all-zeros


# model
print('modeling')

def create_model():
    model = Sequential()
    model.add(lrs.Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False))
    model.add(lrs.SpatialDropout1D(0.2)) 
    model.add(lrs.Bidirectional(lrs.LSTM(128, return_sequences=True, dropout=0.0, recurrent_dropout=0.0), 
        merge_mode='concat')) 
    model.add(lrs.Conv1D(64, kernel_size=2, padding='valid', kernel_initializer='glorot_uniform'))
    model.add(lrs.GlobalMaxPooling1D())    # avg pooling
    model.add(lrs.Dense(6, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=0.001),metrics=['accuracy']) # default 0.002
    return model

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC_AUC - epoch {:d} - score {:.6f}".format(epoch, score))
        return score

def get_calls(x_val, y_val, weights):
    ra_val = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
    checkpoint = ModelCheckpoint(weights, monitor='val_acc', verbose=1, save_best_only=True, mode='max')  # val_acc
    early = EarlyStopping(monitor='val_acc', mode='max', patience=5) # val_acc
    callbacks_list = [ra_val, checkpoint, early]
    return callbacks_list

def fit_model(x_tra, y_tra, x_val, y_val, weights):
    clbks = get_calls(x_val, y_val, weights)
    clf = create_model()
    clf.fit(x_tra, y_tra, batch_size=128, epochs=4, validation_data=(x_val, y_val),     #######
            callbacks=clbks, verbose=1)
    clf.load_weights(weights)
    return clf


# In[ ]:



# do it
splits = 8                                         #$#####
testpreds_list = []
trainpreds_list = []
valindex_list = []
skf = StratifiedKFold(n_splits=splits, shuffle=True)
for i, (train_index, val_index) in enumerate(skf.split(np.zeros(train.shape[0]), y_multi)):
    print ("\n\n\n Training on fold {} \n\n\n".format(str(i+1)))
    xt, xv = x_train[train_index], x_train[val_index]
    yt, yv = y_train[train_index], y_train[val_index]
    weights_file = "./models/weights_best_{}.hdf5".format(str(i))
    clfr = fit_model(xt, yt, xv, yv, weights_file)
    trainpreds_list.append(clfr.predict(xv, batch_size=1024, verbose=1))
    testpreds_list.append(clfr.predict(x_test, batch_size=1024, verbose=1))
    valindex_list.append(val_index)
train_preds = trainpreds_list
test_preds = sum(testpreds_list)/len(testpreds_list)


# In[ ]:


val_indices = np.concatenate(valindex_list, axis=0)


# In[ ]:


trainpreds_all = np.concatenate(trainpreds_list, axis=0)


# In[ ]:


predstrain =  pd.DataFrame(data=trainpreds_all, index=val_indices, columns=class_names)
predstrain.sort_index(inplace=True)


# In[ ]:


predfile =  pd.DataFrame.from_dict({'id': train['id']})
predfile[class_names] = predstrain
predfile.to_csv("../ensembles/preds_bilstmft.csv", index = False)



submission = pd.read_csv("../input/sample_submission.csv")
submission[class_names] = (test_preds)
submission.to_csv("../ensembles/test_bilstmft.csv", index = False)

