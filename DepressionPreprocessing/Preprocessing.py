import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import string
import re


#read the dataset into df
cols_names = ["id","target","tweet"]
df = pd.read_csv('tweets.csv', usecols=cols_names)

#extract tweets column and prepare a sample of 10k tweets for preprocessing
tweets = df["tweet"]
tweets_sample = tweets[:10000]


#tokenize tweets using nltk TweetTokenizer, remove user handles and reduce length
tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

tokenized_tweets = []

for i in range(len(tweets_sample)):
    tokenized_tweets.append(tweet_tokenizer.tokenize(tweets_sample[i]))
    

print("After TweetTokenization")
print(tokenized_tweets[340])

#list of stopwords to remove from tweets tokens
stop_words = set(stopwords.words('english'))


#remove urls, punctuation, and stopwords
for tokenized_tweet in tokenized_tweets:
    for token in tokenized_tweet:
        if token in stop_words:
            tokenized_tweet.remove(token)
        url = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", ' ', token)
        if url == ' ':
            tokenized_tweet.remove(token)
            
for tokenized_tweet in tokenized_tweets:
    for token in tokenized_tweet:
        if token in string.punctuation:
            tokenized_tweet.remove(token)

print("After removing stopwords, punctuation, and urls")
print(tokenized_tweets[340])

#Normalize tokenized tweets using Porter Stem
porter_stem = PorterStemmer()

for tweet_index in range(len(tokenized_tweets)):
    for token_index in range(len(tokenized_tweets[tweet_index])):
        tokenized_tweets[tweet_index][token_index] = porter_stem.stem(tokenized_tweets[tweet_index][token_index])
        

print("After normalization using porter stem")
print(tokenized_tweets[340])

target_tweet_only = {"target":df["target"][:10000], "tweet_tokenized":tokenized_tweets.copy()}
df_target_tweet = pd.DataFrame(target_tweet_only)
print(df_target_tweet.head())

        
df_target_tweet.to_csv('test.csv', index=False, columns = ["target","tweet_tokenized"])



print(" ".join(tokenized_tweets[0]))














