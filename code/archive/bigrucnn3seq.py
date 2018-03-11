
import numpy as np
import pandas as pd

# import string
# import re

from time import datetime

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
maxlength = 150         #150
embed_size = 300

embedding_file = "../vecs/glove.840B.300d.txt"     #####
weights_file = "./weights/weights_best.hdf5"
sub_file = "../input/sample_submission.csv"


# preprocess
print('preprocessing')

# def char_reduce(text):
#     for ch in string.ascii_lowercase[:27]:
#         if ch in text:
#             template = r"("+ch+")\\1{2,}"
#             text = re.sub(template, ch, text)
#     return text

class_names = list(train)[-6:]
y_train = train[class_names].values

train['comment_text'].fillna("no comment")
test['comment_text'].fillna("no comment")



# x_train0 = train['comment_text'].str.replace('?', ' ? ')
# x_train0 = train['comment_text'].str.replace('!', ' ! ')
# x_train0 = x_train0.apply(char_reduce)


# x_test0 = test['comment_text'].str.replace('?', ' ? ')   # currently filtered out by keras tokenizer
# x_test0 = test['comment_text'].str.replace('!', ' ! ')
# x_test0 = x_test0.apply(char_reduce)

x_train0 = train['comment_text']
x_test0 = test['comment_text']

toker = text.Tokenizer(num_words=max_features, 
    filters='?!"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n',
    lower=True)
toker.fit_on_texts(list(x_train0)+list(x_test0))

x_train = toker.texts_to_sequences(x_train0)
x_test = toker.texts_to_sequences(x_test0)

x_train = sequence.pad_sequences(x_train, maxlen=maxlength)
x_test = sequence.pad_sequences(x_test, maxlen=maxlength)


# prepare word vectors/matrix
print('vectorizing')

embeddings_index = {}
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




# build model
print('building model')

def create_model():
    model = Sequential()
    model.add(lrs.Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False))
    model.add(lrs.SpatialDropout1D(0.2)) # 0.2
    model.add(lrs.Bidirectional(lrs.LSTM(128, return_sequences=True, dropout=0.0, recurrent_dropout=0.0), # 128 0,0 
        merge_mode='concat'))          # 'concat' 
    model.add(lrs.Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform"))
    model.add(lrs.GlobalMaxPooling1D())    # avg pooling
    model.add(lrs.Dense(6, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=0.001),metrics=['accuracy']) # default 0.002
    model.summary()
    return model

clf = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)


# make callbacks
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.clf.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC_AUC - epoch {:d} - score {:.6f}".format(epoch+1, score))

checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
ra_val = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
callbacks_list = [ra_val, checkpoint, early]



# fit and predict
print('fitting')
x_tra, x_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)
clf.fit(x_tra, y_tra, batch_size=128, epochs=5, validation_data=(x_val, y_val), 
          callbacks=callbacks_list, verbose=1)

print('predicting')
clf.load_weights(weights_file)
y_pred = clf.predict(x_test, batch_size=1024, verbose=1)

submission = pd.read_csv(sub_file)
submission[class_names] = (y_pred)
submission.to_csv("../subs/sub_bigrulstm04_{}.csv".format(datetime.now().strftime('%d_%H_%M')), index = False)

