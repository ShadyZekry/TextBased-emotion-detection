import numpy as np
from tensorflow.keras.utils import to_categorical
from utilities import  save_train_labels, save_val_labels
from tqdm import tqdm
import pandas as pd

train_df = pd.read_csv('./result/OSACT2022-sharedTask-train.csv', usecols=['off_label']).dropna()
val_df = pd.read_csv('./result/OSACT2022-sharedTask-dev.csv', usecols=['off_label']).dropna()

train_df.loc[train_df['off_label'] == 'OFF', 'label'] = 1
train_df.loc[(train_df['off_label'] == 'NOT_OFF'), 'label'] = 0
train_df['label'] = train_df['label'].astype(np.float64)

val_df.loc[val_df['off_label'] == 'OFF', 'label'] = 1
val_df.loc[(val_df['off_label'] == 'NOT_OFF'), 'label'] = 0
val_df['label'] = val_df['label'].astype(np.float64)

y_train = train_df['label'].values
y_train = to_categorical(y_train, num_classes=2)

y_val = val_df['label'].values
y_val = to_categorical(y_val, num_classes=2)

off_count = train_df.loc[train_df['off_label'] == 'OFF'].shape[0]
print(f'Offensive tweets in train: {off_count}')

non_off_count = train_df.loc[train_df['off_label'] == 'NOT_OFF'].shape[0]
print(f'Non offensive tweets in train: {non_off_count}')

off_count = val_df.loc[val_df['off_label'] == 'OFF'].shape[0]
print(f'Offensive tweets in val: {off_count}')

non_off_count = val_df.loc[val_df['off_label'] == 'NOT_OFF'].shape[0]
print(f'Non offensive tweets in val: {non_off_count}')

save_train_labels(y_train, 'offensive')
save_val_labels(y_val, 'offensive')

print('Saved offensive labels successfuly')