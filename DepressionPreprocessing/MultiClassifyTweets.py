import pandas as pd

processed_tweets_df = pd.read_csv(
    './Result/processed_tweets.csv', usecols=['username', 'target'], encoding='utf-8')

users_df = processed_tweets_df.loc[processed_tweets_df['target'] == 1].drop_duplicates(
    subset=['username'], keep='first')

features_df = pd.read_csv('./Result/features.csv',
                          encoding='utf-8', usecols=['words', 'weights'])

classes_dict = {'high': 1, 'medium': 2, 'low': 3}
users = []
tweets = []
target = []

for user in users_df['username']:
    user_tweets_df = pd.read_csv(
        f'./Result/Docs/{user}.csv', encoding='utf-8', usecols=['processed_tweet'])
    for tweet in user_tweets_df['processed_tweet']:
        tweets.append(tweet)
        users.append(user)

target = [0] * len(tweets)
    
user_tweets_df = pd.DataFrame({'users':users, 'tweets':tweets, 'target':target})

target_col_indx = user_tweets_df.columns.get_loc('target')

for tweet_index in range(user_tweets_df['tweets'].shape[0]):
    low = 0
    medium = 0
    high = 0
    for word_index in range(features_df['words'].shape[0]):
        if features_df['words'][word_index] in str(user_tweets_df['tweets'][tweet_index]):
            word_weight = features_df['weights'][word_index]
            if word_weight >= 1 and word_weight <= 3.9:
                high += 1
            elif word_weight >= 4 and word_weight <= 6.9:
                medium += 1
            else:
                low += 1
    if low > medium and low > high:
        user_tweets_df.iat[tweet_index, target_col_indx] = classes_dict['low']
    elif medium > low and medium > high:
        user_tweets_df.iat[tweet_index, target_col_indx] = classes_dict['medium']
    else:
        user_tweets_df.iat[tweet_index, target_col_indx] = classes_dict['high']

user_tweets_df.to_csv('./Result/classified_user_tweets.csv', encoding='utf-8', index=False, columns=['users','tweets','target']) 
