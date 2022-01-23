import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

dataset_df = pd.read_csv('./Result/classified_user_tweets.csv',
                         encoding='utf-8', usecols=['target', 'username'])
features_df = pd.read_csv('./Result/features.csv',
                          encoding='utf-8', usecols=['term', 'weight'])
classified_docs_df = pd.read_csv('./Result/classified_user_docs.csv', encoding='utf-8', usecols=['target', 'doc'])

tfidf_vectorizer = TfidfVectorizer(input='content', encoding='utf-8', decode_error='ignore', max_df=classified_docs_df['target'].loc[classified_docs_df['target'] > 0].shape[0], vocabulary=features_df['term'].values, sublinear_tf=True)

tfidf_matrix = tfidf_vectorizer.fit_transform(classified_docs_df['doc'].values)

features_index = tfidf_matrix[0, :].nonzero()[1]
tfidf_scores = zip(features_index, [tfidf_matrix[0, x] for x in features_index])
feature_names = tfidf_vectorizer.get_feature_names()

numerical_dataset_dict = {'target' : classified_docs_df['target'].values, 'features': features_df['term'].values, 'numerical_data':[]}

for doc_index in range(classified_docs_df['doc'].shape[0]):
    feature_vector = []
    for feature_index in range(features_df['term'].shape[0]):
        tfidf_score = tfidf_matrix[doc_index, feature_index]
        feature_vector.append(tfidf_matrix[doc_index, feature_index])
    numerical_dataset_dict['numerical_data'].append(feature_vector)

print('Writing numerical dataset to csv started')
with open('./Result/numerical_docs.csv', 'w') as file:
    file_header = 'target'
    for feature in numerical_dataset_dict['features']:
        file_header += f',{feature}'
    file_header += '\n'
    file.write(file_header)
    for record_ptr in range(len(numerical_dataset_dict['target'])):
        target = numerical_dataset_dict['target'][record_ptr]
        record_line = f'{target}'
        for feature_value in numerical_dataset_dict['numerical_data'][record_ptr]:
            record_line += f',{feature_value}'
        record_line += '\n'
        file.write(record_line)
print('Writing numerical dataset to csv end')


        
        




