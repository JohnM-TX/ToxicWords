
# coding: utf-8

# In[2]:


# %load textprep.py
import re
import string
import os
import psutil
import gc

import numpy as np
import pandas as pd

from collections import defaultdict

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file



# In[3]:



cont_patterns = [
        (b'US', b'United States'),
        (b'IT', b'Information Technology'),
        (b'(W|w)on\'t', b'will not'),
        (b'(C|c)an\'t', b'can not'),
        (b'(I|i)\'m', b'i am'),
        (b'(A|a)in\'t', b'is not'),
        (b'(\w+)\'ll', b'\g<1> will'),
        (b'(\w+)n\'t', b'\g<1> not'),
        (b'(\w+)\'ve', b'\g<1> have'),
        (b'(\w+)\'s', b'\g<1> is'),
        (b'(\w+)\'re', b'\g<1> are'),
        (b'(\w+)\'d', b'\g<1> would')
    ]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

def prepare_for_char_n_gram(text):
  clean = bytes(text.lower(), encoding="utf-8") #lower case
  clean = clean.replace(b"\n", b" ")   # spaces and tabs
  clean = clean.replace(b"\t", b" ")
  clean = clean.replace(b"\b", b" ")
  clean = clean.replace(b"\r", b" ")
  for (pattern, repl) in patterns:
      clean = re.sub(pattern, repl, clean)  # contractions?
  exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
  clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])  # punctuation
  clean = re.sub(b"\d+", b" ", clean)  #numbers
  clean = re.sub(b'\s+', b' ', clean)   # spaces
  clean = re.sub(b" ", b"# #", clean)   # add # signs?
  clean = re.sub(b'\s+$', b'', clean) # ending spaces
  return str(clean, 'utf-8')
   
def count_regexp_occ(regexp="", text=None):
    return len(re.findall(regexp, text))

def perform_nlp(df):     # Check all sorts of content
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    df["nb_nigger"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ
        (r"\d{2}|:\d{2}", x))
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ
    (r"\D\d{2}:\d{2},\d{1,2} \w+ \d{4}", x))
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ
        (r"\D\d{1,2} \w+ \d{4}", x))
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ
        (r"http[s]{0,1}://\S+", x))
    df["has_mail"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\S+\@\w+\.\w+", x))
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ
        (r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ
        (r"\"{4}\S+\"{4}", x))
    
    ip_regexp = r"""(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.)
        {3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"""
    # Now clean comments
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) /            df["clean_comment"].apply(lambda x: 1 + min(99, len(x)))

    # Get the exact length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))


def char_analyzer(text):
  tokens = text.split()
  return [token[i: i+3] for token in tokens for i in range(len(token) - 2)]

gc.enable()

print('reading data')
train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

class_names = list(train)[-6:]


perform_nlp(train)
print('nlp train done')
perform_nlp(test)
print('nlp test done')

train_text = train['clean_comment']
test_text = test['clean_comment']
all_text = pd.concat([train_text, test_text])


print("Creating numerical features")
num_features = [f_ for f_ in train.columns
                if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars", 
                    'has_ip_address'] + class_names]


skl = MinMaxScaler()

train_num_features =(skl.fit_transform(train[num_features]))   #csr_matrix
test_num_features = (skl.fit_transform(test[num_features]))

print("Tfidf on word")
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=50000)                 #20000
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
del word_vectorizer
gc.collect()


print("Tfidf on char n_gram")
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    tokenizer=char_analyzer,
    analyzer='word',
    ngram_range=(1, 1),
    max_features=80000)                    #50000
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
    
print((train_char_features>0).sum(axis=1).max())


gc.collect()


print("Stacking matrices")
csr_trn = hstack(
    [train_char_features, 
    train_word_features, 
    train_num_features],
  format='csr') #.tocsr()

csr_sub = hstack([test_char_features, 
        test_word_features, 
        test_num_features], 
  format='csr')


# del train_text
# del test_text
# del train_word_features
# del test_word_features
# del train_char_features
# del test_char_features
# del train_num_features
# del test_num_features
# gc.collect()




# In[18]:


for i,c in enumerate(class_names):
    y = train[class_names[i]]
    fname = "../input/trainsparse{}.txt".format(c)
    dump_svmlight_file(csr_trn, y, fname, zero_based=True, comment=None, query_id=None, multilabel=False)


# In[16]:


y_fake = np.zeros(test.shape[0])
dump_svmlight_file(csr_sub, y_fake, '../input/testsparse.txt', zero_based=True, comment=None, query_id=None, multilabel=False)

