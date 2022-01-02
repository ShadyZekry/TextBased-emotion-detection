import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
dataset_df = pd.read_csv('./Result/classified_user_tweets.csv',
                         encoding='utf-8', usecols=['target', 'processed_tweet', 'username'])
features_df = pd.read_csv('./Result/features.csv',
                          encoding='utf-8', usecols=['term', 'weight'])

depressed_users = dataset_df['username'].loc[dataset_df['target'] > 0].drop_duplicates().dropna()

idf_vectorizer = TfidfVectorizer(input='filename', encoding='utf-8', decode_error='ignore', max_df=depressed_users.shape[0])

depressed_users_docs = []

for user in depressed_users:
    depressed_users_docs.append(f'./Result/Docs/Depressed/{user}.txt')

idf_vectorizer.fit_transform(depressed_users_docs)
numerical_dataset_dict = {'target' : dataset_df['target'].values, 'features': features_df['term'].values, 'numerical_data':[]}

print('Computing numerical data started')
for tweet in dataset_df['processed_tweet']:
    tweet_tokens = str(tweet).split(' ')
    feature_vector = []
    for feature in features_df['term']:
        tf = 1
        for token in tweet_tokens:
            if token == feature:
                tf += 1
        idf = idf_vectorizer.idf_[idf_vectorizer.vocabulary_[feature]]
        tfidf = (1 + math.log(tf)) * idf
        feature_vector.append(tfidf)
    numerical_dataset_dict['numerical_data'].append(feature_vector.copy())
    feature_vector.clear()
print('Computing numerical data ended')

print('Writing numerical dataset to csv started')
with open('./Result/numerical_dataset.csv', 'w') as file:
    file_header = 'target'
    for feature in numerical_dataset_dict['features']:
        file_header += f',{feature}'
    file_header += '\n'
    file.write(file_header)
    for record_ptr in range(len(numerical_dataset_dict['target'])):
        target = numerical_dataset_dict['target'][record_ptr]
        record_line = f',{target}'
        for feature_value in numerical_dataset_dict['numerical_data'][record_ptr]:
            record_line += f',{feature_value}'
        record_line += '\n'
        file.write(record_line)
print('Writing numerical dataset to csv end')


        
        




