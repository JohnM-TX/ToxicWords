import pandas as  pd
import numpy as np

sub1 = pd.read_csv('../subs/sub_GRU2.csv')
sub2 = pd.read_csv('../subs/sub_GRU3.csv')

subbl = sub1.copy()
classes = list(subbl)[-6:]

subbl[classes] = 0.5*sub1[classes] + 0.5*sub2[classes]

subbl.to_csv('../subs/sub_gru2_gru3.csv', index=False)