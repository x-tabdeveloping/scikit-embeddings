from skembeddings.error import NotInstalled
from skembeddings.tokenization.dummy import DummyTokenizer
from skembeddings.tokenization.filter import TokenFilter

try:
    from skembeddings.tokenization.spacy import SpacyTokenizer
except ModuleNotFoundError:
    SpacyTokenizer = NotInstalled("SpacyTokenizer", "spacy")

try:
    from skembeddings.tokenization.huggingface import HuggingFaceTokenizer
except ModuleNotFoundError:
    HuggingFaceTokenizer = NotInstalled("HuggingFaceTokenizer", "transformers")

__all__ = [
    "DummyTokenizer",
    "TokenFilter",
    "HuggingFaceTokenizer",
    "SpacyTokenizer",
]
