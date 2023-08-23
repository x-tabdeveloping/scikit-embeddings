import importlib
import string
from typing import Iterable, Union

from sklearn.base import BaseEstimator, TransformerMixin

punct_without_dot = "\"#$%&'()*+,-/:<=>@[\\]^_`{|}~"
punct_to_space = str.maketrans(
    string.punctuation, " " * len(string.punctuation)
)
punct_to_space_keep_dots = str.maketrans(
    punct_without_dot, " " * len(punct_without_dot)
)
digit_to_space = str.maketrans(string.digits, " " * len(string.digits))


class DummyTokenizer(TransformerMixin, BaseEstimator):
    """Language agnostic dummy preprocessor that is probably orders
    of magnitudes faster than spaCy but also produces results of
    lower quality.

    Parameters
    ----------
    stop_words: str or iterable of str or None, default None
        Words to remove from all texts.
        If a single string, it is interpreted as a language code,
        and stop words are imported from spaCy.
        If its an iterable of strings, every token will be removed that's
        in the list.
        If None, nothing gets removed.
    lowercase: bool, default True
        Inidicates whether tokens should be lowercased.
    remove_digits: bool, default True
        Inidicates whether digits should be removed.
    remove_punctuation: bool, default True
        Indicates whether the component should remove
        punctuation.
    """

    def __init__(
        self,
        stop_words: Union[str, list[str], None],
        lowercase: bool = True,
        remove_digits: bool = True,
        remove_punctuation: bool = True,
        n_jobs: int = 1,
        chunksize: int = 100,
    ):
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.remove_digits = remove_digits
        self.remove_punctuation = remove_punctuation
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        if isinstance(stop_words, str):
            lang = stop_words
            self.stop_word_set = importlib.import_module(
                f"spacy.lang.{lang}.stop_words"
            ).STOP_WORDS
        elif stop_words is None:
            self.stop_word_set = set()
        else:
            self.stop_word_set = set(stop_words)

    def fit(self, X, y=None):
        """Exists for compatiblity, doesn't do anything."""
        return self

    def partial_fit(self, X, y=None):
        """Exists for compatiblity, doesn't do anything."""
        return self

    def process_string(self, text: str) -> list[str]:
        # Removes digits if asked to
        if self.remove_digits:
            text = text.translate(digit_to_space)
        if self.remove_punctuation:
            text = text.translate(punct_to_space)
        if self.lowercase:
            text = text.lower()
        text = text.strip()
        res = []
        for token in text.split():
            if token not in self.stop_word_set:
                res.append(token)
        return res

    def transform(self, X: Iterable[str]) -> list[list[str]]:
        """Preprocesses document with a dummy pipeline."""
        res = map(self.process_string, X)
        return list(res)

    def get_feature_names_out(self, input_features=None):
        return None
