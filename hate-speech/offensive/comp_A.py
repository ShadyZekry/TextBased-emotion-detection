from utilities import load_train_features, load_val_features, load_train_labels, load_val_labels, bert_tokenize, get_embeddings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
from tqdm import tqdm

x_train = load_train_features('offensive')
y_train = load_train_labels('offensive')


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, y_train)

test_df = pd.read_csv('./dataset/OSACT2022-sharedTask-test-tweets.csv', encoding='utf-8', usecols=['id','tweet'])

test_tweets_texts = test_df['tweet'].values

batch_size = 64
num_batches = test_df.shape[0] // batch_size
batches_rem = test_df.shape[0] % batch_size

test_features = np.empty(shape=(test_df.shape[0], 768), dtype=np.float32)
for batch_step in tqdm(range(1, num_batches), desc='getting features..'):
    i = batch_step - 1
    tokens = bert_tokenize(list(test_tweets_texts[i * batch_size: batch_step * batch_size]))
    test_features[i * batch_size : batch_step * batch_size] = get_embeddings(tokens)

if batches_rem != 0:
    tokens = bert_tokenize(list(test_tweets_texts[test_tweets_texts.shape[0] - batches_rem: ]))
    test_features[test_tweets_texts.shape[0] - batches_rem: ] = get_embeddings(tokens)

test_features = scaler.transform(test_features)


offensive_lr = pickle.load(open('./lr_offensive_model.sav', 'rb'))
lr_predictions = offensive_lr.predict(test_features)

with open('./CHILLAX_subtaskA_1.txt', 'w') as submit_file:
    for prediction in lr_predictions:
        if prediction == 0:
            submit_file.write('NOT_OFF\n')
        elif prediction == 1:
            submit_file.write('OFF\n')

offensive_rf = pickle.load(open('./rf_offensive_model.sav', 'rb'))
rf_predictions = offensive_rf.predict(test_features)

with open('./CHILLAX_subtaskA_2.txt', 'w') as submit_file2:
    for prediction in rf_predictions:
        if prediction == 0:
            submit_file2.write("NOT_OFF\n")
        elif prediction == 1:
            submit_file2.write("OFF\n")
            
print('done and saved successfuly')





