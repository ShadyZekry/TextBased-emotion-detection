import pandas as pd

preprocessed_dataset_df = pd.read_csv(
    './Result/processed_tweets.csv', usecols=['target', 'processed_tweet', 'username'])
features_df = pd.read_csv('./Result/features.csv', usecols=['term', 'weight'])

depressed_tweets_df = preprocessed_dataset_df['processed_tweet'].loc[preprocessed_dataset_df['target'] == 1]


multiclass_targets = []
for _ in preprocessed_dataset_df['processed_tweet']:
    multiclass_targets.append(0)

# algorithm used to multi classify tweets into 3 classes H, M, and L
# H => 1, M => 2, L => 3, NonDepressed => 0
targets_list_indx = 0
for depressed_tweet in depressed_tweets_df:
    tweet_words = str(depressed_tweet).split(' ')
    h = 0
    m = 0
    l = 0
    for word in tweet_words:
        if word in features_df['term'].values:
            word_weight = features_df['weight'].loc[features_df['term']
                                                    == word].values[0]
            print(word, ' word')
            print(word_weight, ' weight')
            if word_weight >= 1 and word_weight <= 3.9:
                h += 1
            elif word_weight >= 4 and word_weight <= 6.9:
                m += 1
            else:
                l += 1
    max_val = max(h, m, l)
    if max_val == h:
        multiclass_targets[targets_list_indx] = 1
    elif max_val == m:
        multiclass_targets[targets_list_indx] = 2
    else:
        multiclass_targets[targets_list_indx] = 3
    targets_list_indx += 1

multiclassified_tweets_dict = {'target': multiclass_targets, 'processed_tweet':
                               preprocessed_dataset_df['processed_tweet'].values, 'username': preprocessed_dataset_df['username'].values}
multiclassified_tweets_df = pd.DataFrame(multiclassified_tweets_dict)
multiclassified_tweets_df.to_csv(
    './Result/classified_user_tweets.csv', encoding='utf-8', index=False)
