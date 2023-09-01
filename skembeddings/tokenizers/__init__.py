from typing import Any, Iterable, Optional

from confection import registry

from skembeddings.error import NotInstalled
from skembeddings.tokenizers._huggingface import (
    BPETokenizer,
    UnigramTokenizer,
    WordLevelTokenizer,
    WordPieceTokenizer,
)

try:
    from skembeddings.tokenizers.spacy import SpacyTokenizer
except ModuleNotFoundError:
    SpacyTokenizer = NotInstalled("SpacyTokenizer", "spacy")


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


@registry.tokenizers.register("spacy_tokenizer.v1")
def make_spacy_tokenizer(
    model: str = "en_core_web_sm",
    patterns: Optional[list[list[dict[str, Any]]]] = None,
    out_attrs: Iterable[str] = ("NORM",),
):
    return SpacyTokenizer(model, patterns, out_attrs)


__all__ = [
    "SpacyTokenizer",
    "BPETokenizer",
    "UnigramTokenizer",
    "WordLevelTokenizer",
    "WordPieceTokenizer",
]
