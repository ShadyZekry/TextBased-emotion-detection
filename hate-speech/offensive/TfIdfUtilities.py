import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfUtilities:
    def __init__(self, dataset_df: pd.DataFrame, text_col: str, vocab_max_size: int, max_df: float, min_df: float, ngram_range: tuple, vocab):
        self.__dataset = dataset_df
        self.__text_col = text_col
        if type(vocab) == type(None):
            self.__vectorizer = TfidfVectorizer(input='content', lowercase=False, max_df=max_df, token_pattern=r"(?u)\b\w\w+\b", min_df=min_df, ngram_range=ngram_range, max_features=vocab_max_size)
        else:
            self.__vectorizer =  TfidfVectorizer(input='content', lowercase=False, max_df=max_df, token_pattern=r"(?u)\b\w\w+\b", min_df=min_df, ngram_range=ngram_range, vocabulary=vocab)

    def extract_features(self) -> pd.DataFrame:
        features = self.__vectorizer.fit_transform(self.__dataset[self.__text_col])
        return pd.DataFrame(features.todense(), columns=self.__vectorizer.get_feature_names_out())
    
    def get_vocab(self) -> np.ndarray:
        return self.__vectorizer.get_feature_names_out()

