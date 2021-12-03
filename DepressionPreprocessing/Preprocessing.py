from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import time
import re
import string
import pandas as pd
from Utils import time_convert

#the column names used in the dataset
col_names = ["target", "tweet", 'username']

#read the dataset from the csv
dataset_df = pd.read_csv('./Dataset/TwitterDataset.csv', encoding='UTF-8', usecols=col_names)

#set of stopwords in english to remove from tweets
stopwords = set(stopwords.words('english'))

#punctuation to remove from tweets
punctuation = string.punctuation

#regular expression to match urls for removal from tweets
urls_re = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

#grab the tweets from the dataset
tweets = dataset_df["tweet"].copy(True)

#nltk tweet tokenizer to tokenize the tweets and remove handles
tokenized_tweets = []
tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)

tweet_tokenizer_start = time.time()
print("Tweet tokenizer started")
for i in range(len(tweets)):
    tokenized_tweets.append(tweet_tokenizer.tokenize(tweets[i].lower()))
    tweets[i] = ' '.join(tokenized_tweets[i])
            
print('Tweet tokenizer ended')

tweet_tokenizer_end = time.time()
print("Elapsed time:", time_convert(tweet_tokenizer_end - tweet_tokenizer_start))

cleaning_start = time.time()
print("Removal of urls, stopwords, and punctuation started")

#remove urls that matches urls_re and remove punctuation and remove stopwords

for i in range(len(tweets)):
    url_matches = re.findall(urls_re, tweets[i])
    split_tweet = tweets[i].split()
    for match_tuple in url_matches:
        for match in match_tuple:
            if match in split_tweet:
                split_tweet.remove(match)
    for w_index in range(len(split_tweet)):
        split_tweet[w_index] = split_tweet[w_index].translate(str.maketrans('','', punctuation))
    tweets[i] = ' '.join(split_tweet)
    for word in tweets[i]:
        if word in stopwords:
            tweets[i].replace(word, '')

cleaning_end = time.time() 
print('Removal of urls, stopwords, and punctuation ended')
print('Elapsed time: ', time_convert(cleaning_end - cleaning_start))

#Normalize tokenized tweets using Porter Stem
porter_start = time.time()
porter_stem = PorterStemmer()

print("Porter normalization started")
tokenized_tweets.clear()

for tweet in tweets:
    tokenized_tweets.append(tweet.split())
    
i = 0
for tokenized_tweet in tokenized_tweets:
    stemmed_tweet = ''
    for token in tokenized_tweet:
        stemmed_tweet += f' {porter_stem.stem(token)}'
    tweets[i] = stemmed_tweet
    i += 1
print('Porter normalization ended')
porter_end = time.time()
print("Elapsed time:", time_convert(porter_end - porter_start))

print('Removing digits from tweets started')
digit_remove_start = time.time()
for i in range(len(tweets)):
    tweets[i] = ''.join([j for j in tweets[i] if not j.isdigit()])
digit_remove_end = time.time()
print('Removing digits from tweets ended')
print(f'Elapsed time: {time_convert(digit_remove_end - digit_remove_start)}')

delete_emojis_start = time.time()
print('Deleting emojis from tweet started')
regrex_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
for i in range(len(tweets)):
    tweets[i] = regrex_pattern.sub(r'', tweets[i])
delete_emojis_end = time.time()
print('Deleting emojis from tweet ended')
print(f'Elapsed time: {time_convert(delete_emojis_end - delete_emojis_start)}')

writing_start = time.time()
print('Writing to csv started')
processed_tweets = {'processed_tweet':tweets, 'target':dataset_df['target'], 'username' : dataset_df['username']}

processed_tweets_df = pd.DataFrame(processed_tweets)

processed_tweets_df.to_csv('./Result/processed_tweets.csv', index=False, columns=['target', 'processed_tweet', 'username'], encoding='utf-8')
writing_end = time.time()
print('Writing to csv ended')
print('Elapsed time ', time_convert(writing_end - writing_start))