import numpy as np
from tensorflow.keras.utils import to_categorical
from utilities import  save_train_labels, save_val_labels
from tqdm import tqdm
import pandas as pd

train_df = pd.read_csv('./result/augmented_hatespeech_data.csv', usecols=['label']).dropna()
val_df = pd.read_csv('./dataset/OSACT2022-sharedTask-dev.csv', usecols=['hs_label']).dropna()

train_df['label'] = train_df['label'].astype(np.float32)

val_df.loc[val_df['hs_label'] == 'HS1', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS2', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS3', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS4', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS5', 'label'] = 1
val_df.loc[val_df['hs_label'] == 'HS6', 'label'] = 1
val_df.loc[(val_df['hs_label'] == 'NOT_HS'), 'label'] = 0
val_df['label'] = val_df['label'].astype(np.float32)

y_train = train_df['label'].values
# y_train = to_categorical(y_train, num_classes=2)

y_val = val_df['label'].values
# y_val = to_categorical(y_val, num_classes=2)

hs_count = train_df.loc[train_df['label'] == 1].shape[0]
print(f'hatespeech tweets in train: {hs_count}')

non_hs_count = train_df.loc[train_df['label'] == 0].shape[0]
print(f'Non hatespeech tweets in train: {non_hs_count}')

hs_count = val_df.loc[val_df['label'] == 1].shape[0]
print(f'hatespeech tweets in val: {hs_count}')

non_hs_count = val_df.loc[val_df['label'] == 0].shape[0]
print(f'Non hatespeech tweets in val: {non_hs_count}')

save_train_labels(y_train, 'hatespeech')
save_val_labels(y_val, 'hatespeech')

print('Saved hatespeech labels successfuly')