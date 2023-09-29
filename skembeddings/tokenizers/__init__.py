from typing import Optional

from confection import registry

from skembeddings.tokenizers._huggingface import (
    BPETokenizer,
    UnigramTokenizer,
    WordLevelTokenizer,
    WordPieceTokenizer,
)


@registry.tokenizers.register("wordlevel_tokenizer.v2")
def make_word_level_tokenizer(vocab_size: int = 30000, min_frequency: int = 0):
    return WordLevelTokenizer(vocab_size, min_frequency)


@registry.tokenizers.register("wordpiece_tokenizer.v2")
def make_wordpiece_tokenizer(vocab_size: int = 30000, min_frequency: int = 0):
    return WordPieceTokenizer(vocab_size, min_frequency)


@registry.tokenizers.register("bpe_tokenizer.v2")
def make_bpe_tokenizer(
    vocab_size: int = 30000,
    min_frequency: int = 0,
    max_token_length: Optional[int] = None,
):
    return BPETokenizer(vocab_size, min_frequency, max_token_length)


@registry.tokenizers.register("unigram_tokenizer.v2")
def make_unigram_tokenizer(
    vocab_size=8000,
    shrinking_factor=0.75,
    max_piece_length=16,
    n_sub_iterations=2,
):
    return UnigramTokenizer(
        vocab_size, shrinking_factor, max_piece_length, n_sub_iterations
    )


__all__ = [
    "BPETokenizer",
    "UnigramTokenizer",
    "WordLevelTokenizer",
    "WordPieceTokenizer",
]
