import pandas as pd
from utilities import compute_seq_length, bert_tokenize, get_embeddings, save_train_features, save_val_features, save_train_labels, save_val_labels
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

train_df = pd.read_csv('./result/OSACT2022-sharedTask-train.csv', usecols=['id','tweet','off_label','hs_label']).dropna()
val_df = pd.read_csv('./result/OSACT2022-sharedTask-dev.csv', usecols=['id','tweet','off_label','hs_label']).dropna()

x_train_texts = train_df['tweet'].values
x_val_texts = val_df['tweet'].values

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

sequence_length = compute_seq_length(x_train_texts)

print(f'sequence_length:{sequence_length}')

train_features = np.empty(shape=(x_train_texts.shape[0], sequence_length, 768), dtype=np.float64)
for i in tqdm(range(x_train_texts.shape[0]), desc='getting training features..'):
    tokens = bert_tokenize(x_train_texts[i], sequence_length)
    train_features[i] = get_embeddings(tokens)

val_features = np.empty(shape=(x_val_texts.shape[0], sequence_length, 768), dtype=np.float64)
for i in tqdm(range(x_val_texts.shape[0]), desc='getting validation features..'):
    tokens = bert_tokenize(x_val_texts[i], sequence_length)
    val_features[i] = get_embeddings(tokens)

off_count = train_df.loc[train_df['off_label'] == 'OFF'].shape[0]
print(f'Offensive tweets in train: {off_count}')

non_off_count = train_df.loc[train_df['off_label'] == 'NOT_OFF'].shape[0]
print(f'Non offensive tweets in train: {non_off_count}')

save_train_labels(y_train, 'offensive')
save_val_labels(y_val, 'offensive')
save_train_features(train_features)
save_val_features(val_features)

print(train_features.shape, ' train features shape')
print(train_features[0:5], ' train_features')

print(val_features.shape, ' val_features shape')
print(val_features[0:5], ' val_features')

print(y_train.shape, ' train_labels shape')
print(y_train[0:5], ' train_labels')

print(y_val.shape, ' val_labels shape')
print(y_val[0:5], ' val_labels')

print(f'saved successfuly')