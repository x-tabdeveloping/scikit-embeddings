from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.pre_tokenizers import ByteLevel, Whitespace

from skembeddings.tokenizers.base import HuggingFaceTokenizerBase


class WordPieceTokenizer(HuggingFaceTokenizerBase):
    def _init_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = self.normalizer
        return tokenizer


class WordLevelTokenizer(HuggingFaceTokenizerBase):
    def _init_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = self.normalizer
        return tokenizer


class UnigramTokenizer(HuggingFaceTokenizerBase):
    def _init_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(Unigram())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.normalizer = self.normalizer
        return tokenizer


class BPETokenizer(HuggingFaceTokenizerBase):
    def _init_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.normalizer = self.normalizer
        return tokenizer
