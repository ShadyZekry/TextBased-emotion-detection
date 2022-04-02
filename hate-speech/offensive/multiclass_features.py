import pandas as pd
from utilities import bert_tokenize, get_embeddings, save_train_features, save_val_features
import numpy as np
from tqdm import tqdm

train_df = pd.read_csv('./result/augmented_multiclass_data.csv', usecols=['tweet']).dropna()
val_df = pd.read_csv('./dataset/OSACT2022-sharedTask-dev.csv', usecols=['tweet']).dropna()

x_train_texts = train_df['tweet'].values
x_val_texts = val_df['tweet'].values

batch_size = 64
train_num_batches = x_train_texts.shape[0] // batch_size
train_rem = x_train_texts.shape[0] % batch_size

train_features = np.empty(shape=(x_train_texts.shape[0], 768), dtype=np.float32)
for batch_step in tqdm(range(1, train_num_batches), desc='getting training features..'):
    i = batch_step - 1
    tokens = bert_tokenize(list(x_train_texts[i * batch_size: batch_step * batch_size]))
    train_features[i * batch_size : batch_step * batch_size] = get_embeddings(tokens)

if train_rem != 0:
    tokens = bert_tokenize(list(x_train_texts[x_train_texts.shape[0] - train_rem: ]))
    train_features[x_train_texts.shape[0] - train_rem: ] = get_embeddings(tokens)

val_num_batches = x_val_texts.shape[0] // batch_size
val_rem = x_val_texts.shape[0] % batch_size

val_features = np.empty(shape=(x_val_texts.shape[0], 768), dtype=np.float32)
for batch_step in tqdm(range(1, val_num_batches), desc='getting validation features..'):
    i = batch_step - 1
    tokens = bert_tokenize(list(x_val_texts[i * batch_size: batch_step * batch_size]))
    val_features[i * batch_size: batch_step * batch_size] = get_embeddings(tokens)

if val_rem != 0:
    tokens = bert_tokenize(list(x_val_texts[x_val_texts.shape[0] - val_rem: ]))
    val_features[x_val_texts.shape[0] - val_rem: ] = get_embeddings(tokens)

save_train_features(train_features, 'multiclass')
save_val_features(val_features, 'multiclass')

print(train_features.shape, ' train features shape')
print(train_features[0:5], ' train_features')

print(val_features.shape, ' val_features shape')
print(val_features[0:5], ' val_features')

print(f'saved features successfuly')