import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from empath import Empath
from nltk.corpus import stopwords

users_df = pd.read_csv('./Result/processed_tweets.csv', encoding='utf-8', usecols=['username', 'target'])

users_df = users_df.drop_duplicates(subset=['username']).loc[users_df['target'] == 1]

cv = CountVectorizer()

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

features = {'words':[], 'weights':[]}

for user in users_df['username']:
    user_doc = pd.read_csv(f'./Result/Docs/{user}.csv', encoding='utf-8', usecols=['processed_tweet'])
    word_count_vector = cv.fit_transform(user_doc['processed_tweet'].values.astype('str'))
    tfidf_transformer.fit(word_count_vector)
    i = 0
    for word in cv.get_feature_names():
        features['words'].append(word)
        features['weights'].append(tfidf_transformer.idf_[i])
        i += 1

features_df = pd.DataFrame(features)
features_df.sort_values(by=['weights'], ascending=False, inplace=True)
features_df.drop_duplicates(subset=['words'], keep='first', inplace=True)

empath = Empath()
attributes_classified = []

for word in features_df['words']:
    analysis_dict = empath.analyze(word)
    for attr in analysis_dict:
        if analysis_dict[attr] > 0 and attr not in attributes_classified:
            attributes_classified.append(attr)

stopwords = set(stopwords.words('english'))

for word in features_df['words']:
    if word in stopwords:
        indices = features_df.index[features_df['words'] == word]
        features_df.drop(labels=indices, axis=0, inplace=True)


attributes_classified_df = pd.DataFrame({'categories':attributes_classified})

attributes_classified_df.to_csv('./Result/categories.csv', encoding='utf-8', index=False, columns=['categories'])

features_df.to_csv('./Result/tfidf-result.csv', encoding='utf-8', columns=['words','weights'], index=False)






