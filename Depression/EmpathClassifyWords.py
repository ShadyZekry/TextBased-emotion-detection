import pandas as pd
from empath import Empath
selected_psych_attrs = ['family', 'friends', 'religion', 'death', 'emotional', 'health'
                        , 'sexual', 'positive_emotion', 'negative_emotion', 'anger', 'sadness'
                        , 'suffering', 'nervousness', 'fear', 'pain', 'hate', 'shame']

empath = Empath()

tfidf_result_df = pd.read_csv('./Result/tfidf-result.csv', encoding='utf-8', usecols=['term','tfidf'])

terms_df = tfidf_result_df['term']

classified_terms = []

for term in terms_df:
    classify_result_dict = empath.analyze(term, categories=selected_psych_attrs)
    for key in classify_result_dict:
        if key in selected_psych_attrs and classify_result_dict[key] != 0:
            classified_terms.append(term)
            break

feature_weight_dict = {'term':[], 'weight':[]}

for term in classified_terms:
    tfidf = tfidf_result_df['tfidf'].loc[tfidf_result_df['term'] == term].values[0]
    feature_weight_dict['term'].append(term)
    feature_weight_dict['weight'].append(tfidf)
    
feature_weight_df = pd.DataFrame(feature_weight_dict)
feature_weight_df.to_csv('./Result/features.csv', encoding='utf-8', index=False)