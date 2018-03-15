

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("../input/train.csv")

# create composite class variable for stratification
class_names = list(train)[-6:]
multarray = np.array([100000, 10000, 1000, 100, 10, 1])
y_multi = np.sum(train[class_names].values * multarray, axis=1)

# set splits
splits = 8
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

# produce two lists of ids. each list has n items where n is the 
#    number of folds and each item is a pandas series of indexed id numbers
train_ids = [] 
val_ids = []
for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(train.shape[0]), y_multi)):
    train_ids.append(train.loc[train_idx, 'id'])
    val_ids.append(train.loc[val_idx, 'id'])

# example use
for i in range(splits):
    trainsplit = train[train.id.isin(train_ids[i])]
    valsplit = train[train.id.isin(val_ids[i])]
    # do stuff with the current fold....
    print('Fold {} : Trainset length {} : Valset length {}'.format(i, trainsplit.shape[0], valsplit.shape[0]))


