import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import string
import re
import time


#helper function to format time from seconds
def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  return "{0}:{1}:{2}".format(int(hours),int(mins), sec)

    
#read the dataset into df
cols_names = ["id","target","tweet"]
df = pd.read_csv('tweets.csv', usecols=cols_names)

#extract tweets column and prepare for preprocessing
tweets = df["tweet"]



#tokenize tweets using nltk TweetTokenizer, remove user handles and reduce length
tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

tokenized_tweets = []

tweet_tokenizer_start = time.time()
print("Tweet tokenizer started")
for i in range(len(tweets)):
    tokenized_tweets.append(tweet_tokenizer.tokenize(tweets[i]))
    if i % 100000 == 0:
        print("Tweet tokenizer is " , i / len(tweets) * 100, "% done")

    
tweet_tokenizer_end = time.time()
print("Tweet tokenizer elapsed time:", time_convert(tweet_tokenizer_end - tweet_tokenizer_start))
time.sleep(3)


#list of stopwords to remove from tweets tokens
stop_words = set(stopwords.words('english'))


remove_urls_start = time.time()

i = 0
print("remove urls, punctuation, stopwords started")
#remove urls, punctuation, and stopwords
for tokenized_tweet in tokenized_tweets:
    for token in tokenized_tweet:
        if token in stop_words:
            tokenized_tweet.remove(token)
        url = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", ' ', token)
        if url == ' ':
            tokenized_tweet.remove(token)
    i += 1
    if i % 100000 == 0:
        print("Url and stopwords removal is " , i / len(tweets) * 100, "% done")

i = 0
for tokenized_tweet in tokenized_tweets:
    for token in tokenized_tweet:
        if token in string.punctuation:
            tokenized_tweet.remove(token)
    i += 1
    if i % 100000 == 0:
        print("Punctuation removal is " , i / len(tweets) * 100, "% done")



remove_urls_end = time.time()
print("url, punctuation and stopwords removal elapsed time:", time_convert(remove_urls_end - remove_urls_start))
time.sleep(3)

porter_start = time.time()
#Normalize tokenized tweets using Porter Stem
porter_stem = PorterStemmer()

print("normalization started")

i = 0
for tweet_index in range(len(tokenized_tweets)):
    for token_index in range(len(tokenized_tweets[tweet_index])):
        tokenized_tweets[tweet_index][token_index] = porter_stem.stem(tokenized_tweets[tweet_index][token_index])
    i += 1
    if i % 100000 == 0:
        print("Porter normalization is " , i / len(tweets) * 100, "% done")
        
porter_end = time.time()
print("Normalization elapsed time:", time_convert(porter_end - porter_start))

writing_start = time.time()

for tweet_index in range(len(tokenized_tweets)):
    tokenized_tweets[tweet_index] = " ".join(tokenized_tweets[tweet_index])


target_tweet_only = {"target": df["target"], "tokenized_tweet": tokenized_tweets.copy()}
df_target_tweet = pd.DataFrame(target_tweet_only)

df_target_tweet.to_csv('PreprocessedTweets.csv',index=False, columns=["target", "tokenized_tweet"])
writing_end = time.time()

print("Writing result to csv elapsed time:" , time_convert(writing_end - writing_start))

















