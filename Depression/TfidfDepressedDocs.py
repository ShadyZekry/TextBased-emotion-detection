import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from empath import Empath

dataset_df = pd.read_csv('./Result/processed_tweets.csv',
                         usecols=['target', 'processed_tweet', 'username'])

depressed_users_df = dataset_df['username'].loc[dataset_df['target']
                                                == 1].drop_duplicates().dropna()

depressed_users = depressed_users_df.iloc[0:]

depressed_users_docs = []

for user in depressed_users:
    depressed_users_docs.append(f'./Result/Docs/Depressed/{user}.txt')

tfidf_vectorizer_tf = TfidfVectorizer(input='filename', decode_error='ignore', norm=None, max_features=100, sublinear_tf=True)
tfidf_vectorizer_idf = TfidfVectorizer(input='filename', decode_error='ignore', max_df=len(depressed_users_docs))
tfidf_vectorizer_idf_matrix = tfidf_vectorizer_idf.fit_transform(depressed_users_docs)

tfidf_result_dict = {'term': [], 'tfidf': []}

for doc in depressed_users_docs:
    tfidf_matrix = tfidf_vectorizer_tf.fit_transform([doc])
    feature_names = tfidf_vectorizer_tf.get_feature_names()
    feature_index = tfidf_matrix[0, :].nonzero()[1]
    tfidf_scores = zip(
        feature_index, [tfidf_matrix[0, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        tfidf_result_dict['term'].append(w)
        tfidf = s * tfidf_vectorizer_idf.idf_[tfidf_vectorizer_idf.vocabulary_[w]]
        tfidf_result_dict['tfidf'].append(tfidf)

tfidf_result_df = pd.DataFrame(tfidf_result_dict)
tfidf_result_df = tfidf_result_df.drop_duplicates(keep='first', subset=['term'])
tfidf_result_df.to_csv('./Result/tfidf-result.csv',
                       index=False, encoding='utf-8')

empath = Empath()

categories_analyzed = []

for term in tfidf_result_dict['term']:
    result_dict = empath.analyze(term)
    for key in result_dict:
        if result_dict[key] != 0:
            categories_analyzed.append(key)

categories_dict = {'categories': categories_analyzed}

categories_df = pd.DataFrame(categories_dict)

categories_df.drop_duplicates(inplace=True, keep='first')
categories_df.to_csv('./Result/categories.csv', encoding='utf-8', index=False)
