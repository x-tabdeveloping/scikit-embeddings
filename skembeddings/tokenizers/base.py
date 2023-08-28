from abc import ABC, abstractmethod
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from tokenizers import Tokenizer
from tokenizers.normalizers import BertNormalizer, Normalizer


class HuggingFaceTokenizerBase(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, normalizer: Normalizer = BertNormalizer()):
        self.tokenizer = None
        self.normalizer = normalizer

    @abstractmethod
    def _init_tokenizer(self) -> Tokenizer:
        pass

    def fit(self, X: Iterable[str], y=None):
        self.tokenizer = self._init_tokenizer()
        self.tokenizer.train_from_iterator(X)
        return self

    def partial_fit(self, X: Iterable[str], y=None):
        if self.tokenizer is None:
            self.fit(X)
        else:
            new_tokenizer = self._init_tokenizer()
            new_vocab = new_tokenizer.get_vocab()
            self.tokenizer.add_tokens(new_vocab)
        return self

    def transform(self, X: Iterable[str]) -> list[list[str]]:
        if self.tokenizer is None:
            raise NotFittedError("Tokenizer has not been trained yet.")
        if isinstance(X, str):
            raise TypeError(
                "str passed instead of iterable, did you mean to pass [X]?"
            )
        res = []
        for text in X:
            encoding = self.tokenizer.encode(text)
            res.append(encoding.tokens)
        return res

    def get_feature_names_out(self, input_features=None):
        return None
