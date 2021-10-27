
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer 

preprocessed_df = pd.read_csv('PreprocessedTweets.csv', usecols=["target", "tokenized_tweet"])


preprocessed_tweets = preprocessed_df[pd.notnull(preprocessed_df["tokenized_tweet"])]["tokenized_tweet"].astype("string")


cv = CountVectorizer()

word_count_vector = cv.fit_transform(preprocessed_tweets)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)


df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
 
print(df_idf.sort_values(by=['idf_weights']))











