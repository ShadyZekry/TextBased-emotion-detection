from empath import Empath
import pandas as pd

empath = Empath()

psych_attrs = ['weakness', 'family', 'friends', 'suffering', 'death', 'nervousness', 'sadness', 'horror', 'irritability'
               ,'negative_emotion', 'positive_emotion', 'shame', 'pain', 'love', 'emotional', 'help', 'torment'
               'fear', 'anger', 'body', 'violence', 'hate', 'envy', 'swearing_terms', 'sleep', 'kill', 'dispute']

features_df = pd.read_csv('./Result/tfidf-result.csv', encoding='utf-8', usecols=['words', 'weights'])
features_df.sort_values(by=['weights'], inplace=True, ascending=False)

for word in features_df['words']:
    analysis_dict = empath.analyze(word, categories=psych_attrs)
    belongs = False
    for attr in analysis_dict:
        if analysis_dict[attr] > 0:
            belongs = True
    if not belongs:
        indices = features_df.index[features_df['words'] == word]
        features_df.drop(labels=indices, axis=0, inplace=True)

features_df.to_csv('./Result/features.csv', encoding='utf-8', columns=['words', 'weights'], index=False)