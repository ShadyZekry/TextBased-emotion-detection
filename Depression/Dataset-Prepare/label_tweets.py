import pandas as pd

d_tweets_df = pd.read_csv('d_tweets.csv', usecols=['tweet','language', 'username'])

non_d_tweets_df = pd.read_csv('non_d_tweets.csv', usecols=['tweet','language', 'username'])

d_tweets_df = d_tweets_df.loc[d_tweets_df['language'] == 'en']

non_d_tweets_df = non_d_tweets_df.loc[non_d_tweets_df['language'] == 'en']

depressed_label_col = [1] * d_tweets_df.shape[0]

non_depressed_label_col = [0] * non_d_tweets_df.shape[0]

target_col = pd.concat([pd.Series(depressed_label_col),pd.Series(non_depressed_label_col)], axis=0, join='outer')

tweets_col = list(pd.concat([d_tweets_df['tweet'],non_d_tweets_df['tweet']], axis=0, join='outer'))

username_col = list(pd.concat([d_tweets_df['username'], non_d_tweets_df['username']], axis=0, join='outer'))

dataset_dict = {'target':target_col, 'tweet':tweets_col, 'username': username_col}

dataset_df = pd.DataFrame(dataset_dict)

dataset_df.to_csv('TwitterDataset.csv', encoding='utf-8', columns=['target','tweet', 'username'], index=False)

