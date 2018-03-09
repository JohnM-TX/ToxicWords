
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('autosave', '600')

import time
start_time = time.time()

import sys
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from keras.preprocessing import text, sequence
from keras.layers import Input, Embedding, Dense, Dropout
from keras.layers import SpatialDropout1D, Conv1D, Bidirectional, LSTM
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, Callback


# from keras import initializers, regularizers, constraints
# from keras.layers import , , Activation, 
# from keras.layers import , GlobalMaxPool1D, MaxPooling1D, Add, Flatten
# from keras.layers import , BatchNormalization
# , load_model
# from keras.engine import InputSpec, Layer
# from keras.optimizers import Adam, RMSprop
# from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# from keras.callbacks import Callback


# In[2]:


# get data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


max_features = 100000
maxlength = 150
embed_size = 300

embedding_file = "../vecs/glove.840B.300d.txt"
weights_file = "./weights/weights_best.hdf5"
sub_file = "../input/sample_submission.csv"


# In[4]:


class_names = list(train)[-6:]
y_train = train[class_names].values


#preprocess
train['comment_text'].fillna("no comment")
test['comment_text'].fillna("no comment")

x_train0 = train["comment_text"]
x_test0 = test["comment_text"]

tok = text.Tokenizer(num_words=max_features, lower=True)
tok.fit_on_texts(list(x_train0)+list(x_test0))

x_train = tok.texts_to_sequences(x_train0)
x_test = tok.texts_to_sequences(x_test0)

x_train = sequence.pad_sequences(x_train, maxlen=maxlength)
x_test = sequence.pad_sequences(x_test, maxlen=maxlength)


# In[5]:



embeddings_index = {}
with open(embedding_file, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs          
               

word_index = tok.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


# In[ ]:


sequence_input = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform")(x)
# x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
# x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool]) 
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.1)(x)

preds = Dense(6, activation="sigmoid")(x)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=0.001),metrics=['accuracy']) # default 0.002


# In[ ]:


batch_size = 128
epochs = 4
x_tra, x_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)

checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
ra_val = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
callbacks_list = [ra_val, checkpoint, early]


# In[ ]:


model.fit(x_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), 
          callbacks=callbacks_list, verbose=1)

model.load_weights(weights_file)
print('Predicting....')
y_pred = model.predict(x_test, batch_size=1024, verbose=1)


# In[ ]:



submission = pd.read_csv(sub_file)
submission[class_names] = (y_pred)
submission.to_csv("../subs/sub_bigrucnn.csv", index = False)

print("[{}] from Start to Finish.".format(time.time() - start_time))

