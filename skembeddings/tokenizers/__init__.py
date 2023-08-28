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


__all__ = [
    "SpacyTokenizer",
    "BPETokenizer",
    "UnigramTokenizer",
    "WordLevelTokenizer",
    "WordPieceTokenizer",
]
