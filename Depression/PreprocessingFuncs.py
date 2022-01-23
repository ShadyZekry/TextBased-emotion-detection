from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re


def tokenize(document):
    tweet_tokenizer = TweetTokenizer(strip_handles=True)
    return tweet_tokenizer.tokenize(document)


def remove_urls(tokens):
    urls_re = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    for token in tokens:
        url_matches = re.fullmatch(urls_re, token)
        if url_matches != None:
            tokens.remove(token)
    return tokens


def casefolding(tokens):
    return [t.casefold() for t in tokens]


def remove_stopwords(tokens):
    stop_words = stopwords.words('english')
    return [t for t in tokens if not t in stop_words]


def remove_emojis(tokens):
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
    tokens = [regrex_pattern.sub(r'', token) for token in tokens]
    return tokens


def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def remove_punctuation(tokens):
    puncts = string.punctuation
    puncts += '’'
    puncts += '“'
    puncts += '”'
    puncts += '•'
    translator = str.maketrans('', '', puncts)
    return [t.translate(translator) for t in tokens]


def remove_symbols(tokens):
    return [token for token in tokens if token.encode('utf-8') != '�'.encode('utf-8')]


def remove_digits(tokens):
    digits_str = '0123456789'
    translator = str.maketrans('', '', digits_str)
    return [t.translate(translator) for t in tokens]


def preprocess(document):
    tokens = tokenize(document)
    tokens = casefolding(tokens)
    tokens = remove_urls(tokens)
    tokens = remove_emojis(tokens)
    tokens = remove_stopwords(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_symbols(tokens)
    tokens = remove_digits(tokens)
    tokens = lemmatize(tokens)
    return tokens
