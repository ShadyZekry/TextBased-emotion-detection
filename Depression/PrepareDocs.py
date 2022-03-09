import pandas as pd

features_df = pd.read_csv('./Result/features.csv', encoding='utf-8', usecols=['term', 'weight'])

dataset_df = pd.read_csv('./Dataset/TwitterDataset.csv', encoding='utf-8', usecols=['target', 'username'])

depressed_users_df = dataset_df['username'].loc[dataset_df['target'] > 0].drop_duplicates().dropna()

nondepressed_users_df = dataset_df['username'].loc[dataset_df['target'] == 0].drop_duplicates().dropna()

depressed_users_docs = []
nondepressed_users_docs = []

docs_targets = []

for depressed_user in depressed_users_df:
    with open(f'./Result/Docs/Depressed/{depressed_user}.txt', 'r') as doc:
        depressed_users_docs.append(doc.read())

for nondepressed_user in nondepressed_users_df:
    with open(f'./Result/Docs/NonDepressed/{nondepressed_user}.txt', 'r') as doc:
        nondepressed_users_docs.append(doc.read())
        
for doc in depressed_users_docs:
    docs_targets.append(1)

for doc in nondepressed_users_docs:
    docs_targets.append(0)

classified_docs_dict = {'target':docs_targets, 'doc':depressed_users_docs + nondepressed_users_docs}
classified_docs_df = pd.DataFrame(classified_docs_dict)
classified_docs_df.to_csv('./Result/classified_user_docs.csv', encoding='utf-8', index=False, columns=['target', 'doc'])