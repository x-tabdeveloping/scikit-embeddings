from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.trainers import (
    BpeTrainer,
    Trainer,
    UnigramTrainer,
    WordLevelTrainer,
    WordPieceTrainer,
)

from skembeddings.tokenizers.base import HuggingFaceTokenizerBase


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
