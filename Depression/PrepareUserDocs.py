import pandas as pd
dataset_df = pd.read_csv('./Result/processed_tweets.csv',
                         usecols=['target', 'processed_tweet', 'username'])

depressed_users_df = dataset_df['username'].loc[dataset_df['target'] == 1].drop_duplicates(keep='first')
nondepressed_users_df = dataset_df['username'].loc[dataset_df['target'] == 0].drop_duplicates(keep='first')

depressed_users_docs = {}
nondepressed_users_docs = {}

print("collecting depressed users documents started")
for depressed_user in depressed_users_df:
    print(f'Collecting document for {depressed_user}')
    user_tweets = dataset_df['processed_tweet'].loc[dataset_df['username']
                                          == depressed_user].values
    doc = user_tweets
    depressed_users_docs[depressed_user] = doc
print("collecting depressed users documents ended")

print("Writing depressed users documents started")
for user in depressed_users_docs:
    with open(f'./Result/Docs/Depressed/{user}.txt', 'w') as file:
        for char in depressed_users_docs[user]:
            try:
                file.write(char)
            except:
                continue
print("Writing depressed users documents ended")

print("collecting non depressed users documents started")
for nondepressed_user in nondepressed_users_df:
    print(f'collecting documents for: {nondepressed_user}')
    user_tweets = dataset_df['processed_tweet'].loc[dataset_df['username']
                                          == nondepressed_user].values
    doc = user_tweets
    nondepressed_users_docs[nondepressed_user] = doc
print("collecting non depressed users documents ended")

print("Writing non depressed users documents started")
for user in nondepressed_users_docs:
    with open(f'./Result/Docs/NonDepressed/{user}.txt', 'w') as file:
        for char in nondepressed_users_docs[user]:
            try:
                file.write(char)
            except:
                continue
print("Writing non depressed users documents ended")
