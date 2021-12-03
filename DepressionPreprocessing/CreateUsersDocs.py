import pandas as pd

processed_dataset_df = pd.read_csv('./Result/processed_tweets.csv', encoding='utf-8', usecols=['target', 'processed_tweet', 'username'])

processed_dataset_df = processed_dataset_df.loc[processed_dataset_df['target'] == 1]

users_unique_df = processed_dataset_df.drop_duplicates(subset='username', keep='last', inplace=False)

for user in users_unique_df['username']:
    tweets = processed_dataset_df.loc[processed_dataset_df['username'] == user]
    user_doc = {'processed_tweet':tweets['processed_tweet'], 'username':[user] * tweets.shape[0]}
    doc_df = pd.DataFrame(user_doc)
    doc_df.to_csv(f'./Result/Docs/{user}.csv', encoding='utf-8', index=False, columns=['processed_tweet', 'username'])
    