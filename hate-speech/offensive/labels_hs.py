import numpy as np
from tensorflow.keras.utils import to_categorical
from utilities import  save_train_labels, save_val_labels
from tqdm import tqdm
import pandas as pd

train_df = pd.read_csv('./result/OSACT2022-sharedTask-train.csv', usecols=['hs_label']).dropna()
val_df = pd.read_csv('./result/OSACT2022-sharedTask-dev.csv', usecols=['hs_label']).dropna()

train_df.loc[train_df['hs_label'] == 'HS1', 'label'] = 1
train_df.loc[train_df['hs_label'] == 'HS2', 'label'] = 1
train_df.loc[train_df['hs_label'] == 'HS3', 'label'] = 1
train_df.loc[train_df['hs_label'] == 'HS4', 'label'] = 1
train_df.loc[train_df['hs_label'] == 'HS5', 'label'] = 1
train_df.loc[train_df['hs_label'] == 'HS6', 'label'] = 1
train_df.loc[(train_df['hs_label'] == 'NOT_HS'), 'label'] = 0
train_df['label'] = train_df['label'].astype(np.float64)

val_df.loc[val_df['hs_label'] == 'HS1', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS2', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS3', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS4', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS5', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS6', 'label'] = 1
val_df.loc[(val_df['hs_label'] == 'NOT_HS'), 'label'] = 0
val_df['label'] = val_df['label'].astype(np.float64)

y_train = train_df['label'].values
y_train = to_categorical(y_train, num_classes=2)

y_val = val_df['label'].values
y_val = to_categorical(y_val, num_classes=2)

hs_count = np.count_nonzero(np.argmax(y_train, axis=-1) == 1)
print(f'hs tweets in train: {hs_count}')

non_hs_count = np.count_nonzero(np.argmax(y_train, axis=-1) == 0)
print(f'Non hs tweets in train: {non_hs_count}')

hs_count = np.count_nonzero(np.argmax(y_val, axis=-1) == 1)
print(f'hs tweets in val: {hs_count}')

non_hs_count = np.count_nonzero(np.argmax(y_val, axis=-1) == 0)
print(f'Non hs tweets in val: {non_hs_count}')

save_train_labels(y_train, 'hs')
save_val_labels(y_val, 'hs')

print('Saved hate speech labels successfuly')