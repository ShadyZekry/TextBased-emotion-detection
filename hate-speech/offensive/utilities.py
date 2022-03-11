from tashaphyne.stemming import ArabicLightStemmer
import arabicstopwords.arabicstopwords as stp
import unicodedata as ucd
from pyarabic import araby

stemmer = ArabicLightStemmer()

def check_stopwords(x):
    return not stp.is_stop(x)

def remove_punctuation(x):
    return ''.join(c for c in x if not ucd.category(c).startswith('P'))

def stem_wrapper(token):
    stemmer.light_stem(token)
    return stemmer.get_root()

def preprocess(tweet: str) -> str:
    return ' '.join(araby.tokenize(tweet, morphs=[araby.normalize_alef, araby.normalize_hamza
                                                    , araby.normalize_teh, araby.strip_diacritics, araby.strip_tatweel
                                                    , araby.strip_harakat, remove_punctuation, stem_wrapper], conditions=[araby.is_arabicrange, check_stopwords]))