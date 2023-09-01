from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.normalizers import BertNormalizer, Normalizer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.trainers import (
    BpeTrainer,
    Trainer,
    UnigramTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
)


class PretrainedHuggingFaceTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def fit(self, X: Iterable[str], y=None):
        return self

    def partial_fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> list[list[str]]:
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


class HuggingFaceTokenizerBase(BaseEstimator, TransformerMixin, ABC):
    def __init__(self, normalizer: Normalizer = BertNormalizer()):
        self.tokenizer = None
        self.trainer = None
        self.normalizer = normalizer

    @abstractmethod
    def _init_tokenizer(self) -> Tokenizer:
        pass

    @abstractmethod
    def _init_trainer(self) -> Trainer:
        pass

    def fit(self, X: Iterable[str], y=None):
        self.tokenizer = self._init_tokenizer()
        self.trainer = self._init_trainer()
        self.tokenizer.train_from_iterator(X, self.trainer)
        return self

    def partial_fit(self, X: Iterable[str], y=None):
        if (self.tokenizer is None) or (self.trainer is None):
            self.fit(X)
        else:
            new_tokenizer = self._init_tokenizer()
            new_tokenizer.train_from_iterator(X, self.trainer)
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

    def save(self, path: Union[str, Path]) -> None:
        if self.tokenizer is None:
            raise NotFittedError(
                "Cannot save tokenizer as it has not been fitted yet."
            )
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: Union[str, Path]):
        tokenizer = Tokenizer.from_file(path)
        return PretrainedHuggingFaceTokenizer(tokenizer)


class WordPieceTokenizer(HuggingFaceTokenizerBase):
    def _init_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = self.normalizer
        return tokenizer

    def _init_trainer(self) -> Trainer:
        return WordPieceTrainer(special_tokens=["[UNK]"])


class WordLevelTokenizer(HuggingFaceTokenizerBase):
    def _init_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = self.normalizer
        return tokenizer

    def _init_trainer(self) -> Trainer:
        return WordLevelTrainer(special_tokens=["[UNK]"])


class UnigramTokenizer(HuggingFaceTokenizerBase):
    def _init_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(Unigram())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.normalizer = self.normalizer
        return tokenizer

    def _init_trainer(self) -> Trainer:
        return UnigramTrainer(unk_token="[UNK]", special_tokens=["[UNK]"])


class BPETokenizer(HuggingFaceTokenizerBase):
    def _init_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.normalizer = self.normalizer
        return tokenizer

    def _init_trainer(self) -> Trainer:
        return BpeTrainer(special_tokens=["[UNK]"])
