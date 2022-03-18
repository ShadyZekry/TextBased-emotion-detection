import arabicstopwords.arabicstopwords as stp
import unicodedata as ucd
from pyarabic import araby
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from nltk.stem import ISRIStemmer

marbert_model_path = './marbert-model/'
tokenizer = AutoTokenizer.from_pretrained(marbert_model_path, from_tf=True)
marbert_model = TFAutoModel.from_pretrained(marbert_model_path, output_hidden_states=True)
stemmer = ISRIStemmer()

def check_stopwords(x):
    return not stp.is_stop(x)

def remove_punctuation(x):
    return ''.join(c for c in x if not ucd.category(c).startswith('P'))

def preprocess(tweet: str) -> str:
    preprocessed_tweet = ' '.join(araby.tokenize(tweet, morphs=[remove_punctuation]))
    return preprocessed_tweet

# def compute_seq_length(texts: np.ndarray) -> int:
#     len_dict = {}
#     for text in texts:
#         seq_len = len(tokenizer.tokenize(text))
#         len_dict[seq_len] = 0
#     for text in texts:
#         seq_len = len(tokenizer.tokenize(text))
#         len_dict[seq_len] += 1
#     temp = list(len_dict.keys())[0]
#     for key in len_dict.keys():
#         if len_dict[key] > len_dict[temp]:
#             temp = key
#     return temp + 2

# def bert_tokenize(text: str, seq_length: int) -> list:
#     tokens = tokenizer(text, padding='max_length', truncation=True, max_length=seq_length, add_special_tokens=True)
#     return (tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'])

def bert_tokenize(texts: str) -> list:
    max_len = 0
    for text in texts:
        max_len = max(len(tokenizer.tokenize(text)), max_len)
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len + 2, add_special_tokens=True)
    return (tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'])

def get_embeddings(tokens):
    ids = tf.convert_to_tensor(tokens[0])
    mask = tf.convert_to_tensor(tokens[1])
    type_ids = tf.convert_to_tensor(tokens[2])
    hidden_states = marbert_model(input_ids=ids, attention_mask=mask, token_type_ids=type_ids)[2]
    sentence_embd = tf.reduce_mean(tf.reduce_sum(tf.stack(hidden_states[-4:]), axis = 0), axis=1)
    return sentence_embd

def get_max_length(tweets):
    max_len = 0
    for tweet in tweets:
        split = tweet.split(' ')
        max_len = max(len(split), max_len)
    return max_len

def save_train_features(features: np.ndarray):
    np.save('./train_features.npy', allow_pickle=False, arr=features)

def save_val_features(features: np.ndarray):
    np.save('./val_features.npy', allow_pickle=False, arr=features)

def load_train_features() -> np.ndarray:
    return np.load('./train_features.npy', allow_pickle=False)

def load_val_features() -> np.ndarray:
    return np.load('./val_features.npy', allow_pickle=False)

def save_train_labels(labels: np.ndarray, task_name: str):
    np.save(f'./{task_name}_train_labels.npy', allow_pickle=False, arr=labels)

def save_val_labels(labels: np.ndarray, task_name: str):
    np.save(f'./{task_name}_val_labels.npy', allow_pickle=False, arr=labels)

def load_train_labels(task_name: str) -> np.ndarray:
    return np.load(f'./{task_name}_train_labels.npy', allow_pickle=False)

def load_val_labels(task_name: str) -> np.ndarray:
    return np.load(f'./{task_name}_val_labels.npy', allow_pickle=False)