from typing import Iterable, Optional

from confection import Config, registry
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from tokenizers import Tokenizer
from tokenizers.models import BPE, Model, Unigram, WordLevel, WordPiece
from tokenizers.normalizers import BertNormalizer, Normalizer
from tokenizers.pre_tokenizers import ByteLevel, PreTokenizer, Whitespace
from tokenizers.trainers import (
    BpeTrainer,
    Trainer,
    UnigramTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
)

from skembeddings.base import Serializable
from skembeddings.utils import reusable


def build_tokenizer(
    model: Model,
    pre_tokenizer: PreTokenizer,
    normalizer: Normalizer,
) -> Tokenizer:
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.normalizer = normalizer
    return tokenizer


@reusable
def encode_iterable(
    tokenizer: Tokenizer, X: Iterable[str]
) -> Iterable[list[str]]:
    if isinstance(X, str):
        raise TypeError(
            "str passed instead of iterable, did you mean to pass [X]?"
        )
    for text in X:
        encoding = tokenizer.encode(text)
        yield encoding.tokens


def tok_to_bytes(tokenizer: Tokenizer) -> bytes:
    return tokenizer.to_str().encode("utf-8")


def bytes_to_tok(data: bytes) -> Tokenizer:
    return Tokenizer.from_str(data.decode("utf-8"))


class WordPieceTokenizer(BaseEstimator, TransformerMixin, Serializable):
    def __init__(self, vocab_size: int = 30000, min_frequency: int = 0):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.model_ = None
        self.trainer = WordPieceTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["[UNK]"],
        )

    def fit(self, X: Iterable[str], y=None):
        self.model_ = build_tokenizer(
            WordPiece(unk_token="[UNK]"),
            Whitespace(),
            BertNormalizer(),
        )
        self.model_.train_from_iterator(X, self.trainer)
        return self

    def transform(self, X: Iterable[str]) -> Iterable[list[str]]:
        if self.model_ is None:
            raise NotFittedError(
                "Tokenizer has not been fitted, cannot serialize."
            )
        return encode_iterable(self.model_, X)

    @property
    def config(self) -> Config:
        return Config(
            {
                "tokenizer": {
                    "@tokenizers": "wordpiece_tokenizer.v2",
                    **self.get_params(),
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "WordPieceTokenizer":
        resolved = registry.resolve(config)
        return resolved["tokenizer"]

    def to_bytes(self) -> bytes:
        if self.model_ is None:
            raise NotFittedError(
                "Tokenizer has not been fitted, cannot serialize."
            )
        return tok_to_bytes(self.model_)

    def from_bytes(self, data: bytes):
        self.model_ = bytes_to_tok(data)
        return self


class WordLevelTokenizer(BaseEstimator, TransformerMixin, Serializable):
    def __init__(self, vocab_size: int = 30000, min_frequency: int = 0):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.model_ = None
        self.trainer = WordLevelTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["[UNK]"],
        )

    def fit(self, X: Iterable[str], y=None):
        self.model_ = build_tokenizer(
            WordLevel(unk_token="[UNK]"),
            Whitespace(),
            BertNormalizer(),
        )
        self.model_.train_from_iterator(X, self.trainer)
        return self

    def transform(self, X: Iterable[str]) -> Iterable[list[str]]:
        if self.model_ is None:
            raise NotFittedError(
                "Tokenizer has not been fitted, cannot serialize."
            )
        return encode_iterable(self.model_, X)

    @property
    def config(self) -> Config:
        return Config(
            {
                "tokenizer": {
                    "@tokenizers": "wordlevel_tokenizer.v2",
                    **self.get_params(),
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "WordLevelTokenizer":
        resolved = registry.resolve(config)
        return resolved["tokenizer"]

    def to_bytes(self) -> bytes:
        if self.model_ is None:
            raise NotFittedError(
                "Tokenizer has not been fitted, cannot serialize."
            )
        return tok_to_bytes(self.model_)

    def from_bytes(self, data: bytes):
        self.model_ = bytes_to_tok(data)
        return self


class UnigramTokenizer(BaseEstimator, TransformerMixin, Serializable):
    def __init__(
        self,
        vocab_size=8000,
        shrinking_factor=0.75,
        max_piece_length=16,
        n_sub_iterations=2,
    ):
        self.vocab_size = vocab_size
        self.shrinking_factor = shrinking_factor
        self.max_piece_length = max_piece_length
        self.n_sub_iterations = n_sub_iterations
        self.model_ = None
        self.trainer = UnigramTrainer(
            vocab_size=self.vocab_size,
            shrinking_factor=self.shrinking_factor,
            max_piece_length=self.max_piece_length,
            n_sub_iterations=self.n_sub_iterations,
            special_tokens=["[UNK]"],
        )

    def fit(self, X: Iterable[str], y=None):
        self.model_ = build_tokenizer(
            Unigram(),
            ByteLevel(),
            BertNormalizer(),
        )
        self.model_.train_from_iterator(X, self.trainer)
        return self

    def transform(self, X: Iterable[str]) -> Iterable[list[str]]:
        if self.model_ is None:
            raise NotFittedError(
                "Tokenizer has not been fitted, cannot serialize."
            )
        return encode_iterable(self.model_, X)

    @property
    def config(self) -> Config:
        return Config(
            {
                "tokenizer": {
                    "@tokenizers": "unigram_tokenizer.v2",
                    **self.get_params(),
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "UnigramTokenizer":
        resolved = registry.resolve(config)
        return resolved["tokenizer"]

    def to_bytes(self) -> bytes:
        if self.model_ is None:
            raise NotFittedError(
                "Tokenizer has not been fitted, cannot serialize."
            )
        return tok_to_bytes(self.model_)

    def from_bytes(self, data: bytes):
        self.model_ = bytes_to_tok(data)
        return self


class BPETokenizer(BaseEstimator, TransformerMixin, Serializable):
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 0,
        max_token_length: Optional[int] = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_token_length = max_token_length
        self.model_ = None
        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            max_token_length=self.max_token_length,
            special_tokens=["[UNK]"],
        )

    def fit(self, X: Iterable[str], y=None):
        self.model_ = build_tokenizer(
            BPE(),
            ByteLevel(),
            BertNormalizer(),
        )
        self.model_.train_from_iterator(X, self.trainer)
        return self

    def transform(self, X: Iterable[str]) -> Iterable[list[str]]:
        if self.model_ is None:
            raise NotFittedError(
                "Tokenizer has not been fitted, cannot serialize."
            )
        return encode_iterable(self.model_, X)

    @property
    def config(self) -> Config:
        return Config(
            {
                "tokenizer": {
                    "@tokenizers": "bpe_tokenizer.v2",
                    **self.get_params(),
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "BpeTokenizer":
        resolved = registry.resolve(config)
        return resolved["tokenizer"]

    def to_bytes(self) -> bytes:
        if self.model_ is None:
            raise NotFittedError(
                "Tokenizer has not been fitted, cannot serialize."
            )
        return tok_to_bytes(self.model_)

    def from_bytes(self, data: bytes):
        self.model_ = bytes_to_tok(data)
        return self
