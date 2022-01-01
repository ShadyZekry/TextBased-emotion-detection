from PreprocessingFuncs import preprocess
import pandas as pd

dataset_df = pd.read_csv('./Dataset/TwitterDataset.csv', encoding='utf-8', usecols=['target','tweet','username'])

preprocessed_dataset_dict = {'target':[], 'processed_tweet':[], 'username':[]}

for index, record in dataset_df.iterrows():
    preprocessed_dataset_dict['target'].append(record['target'])
    preprocessed_dataset_dict['username'].append(record['username'])
    tweet = record['tweet']
    preprocessed_tweet = preprocess(tweet)
    preprocessed_dataset_dict['processed_tweet'].append(' '.join(preprocessed_tweet))
    print('Before preprocessing:',end=' ')
    for char in tweet:
        try:
            print(f'{char}', end='')
        except UnicodeEncodeError:
            continue
    print()
    print('After preprocessing:', end=' ')
    for char in ' '.join(preprocessed_tweet):
        try:
            print(f'{char}', end='')
        except UnicodeEncodeError:
            continue
    print()

preprocessed_dataset_df = pd.DataFrame(preprocessed_dataset_dict)
preprocessed_dataset_df.to_csv('./Result/processed_tweets.csv', encoding='utf-8', index=False,columns=['target', 'processed_tweet', 'username'])