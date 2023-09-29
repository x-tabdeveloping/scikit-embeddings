from confection import registry

from skembeddings.tokenizers._huggingface import (
    BPETokenizer,
    UnigramTokenizer,
    WordLevelTokenizer,
    WordPieceTokenizer,
)


@registry.tokenizers.register("word_level_tokenizer.v1")
def make_word_level_tokenizer():
    return WordLevelTokenizer()


@registry.tokenizers.register("wordpiece_tokenizer.v1")
def make_wordpiece_tokenizer():
    return WordPieceTokenizer()


@registry.tokenizers.register("bpe_tokenizer.v1")
def make_bpe_tokenizer():
    return BPETokenizer()


@registry.tokenizers.register("unigram_tokenizer.v1")
def make_unigram_tokenizer():
    return UnigramTokenizer()


__all__ = [
    "BPETokenizer",
    "UnigramTokenizer",
    "WordLevelTokenizer",
    "WordPieceTokenizer",
]
