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

# algorithm used to multi classify tweets into 3 classes H, M, and L
# H => 1, M => 2, L => 3, NonDepressed => 0
for doc in depressed_users_docs:
    tokens = doc.split(' ')
    h = 0
    m = 0
    l = 0
    for word in tokens:
        if word in features_df['term'].values:
            word_weight = features_df['weight'].loc[features_df['term']
                                                    == word].values[0]
            if word_weight >= 1 and word_weight <= 3.9:
                h += 1
            elif word_weight >= 4 and word_weight <= 6.9:
                m += 1
            else:
                l += 1
    max_val = max(h, m, l)
    if max_val == h:
        docs_targets.append(1)
    elif max_val == m:
        docs_targets.append(2)
    else:
        docs_targets.append(3)

for doc in nondepressed_users_docs:
    docs_targets.append(0)

classified_docs_dict = {'target':docs_targets, 'doc':depressed_users_docs + nondepressed_users_docs}
classified_docs_df = pd.DataFrame(classified_docs_dict)
classified_docs_df.to_csv('./Result/classified_user_docs.csv', encoding='utf-8', index=False, columns=['target', 'doc'])