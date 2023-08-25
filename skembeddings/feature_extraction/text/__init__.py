from skembeddings.error import NotInstalled

try:
    from skembeddings.feature_extraction.text.huggingface import \
        HuggingFaceVectorizer
except ModuleNotFoundError:
    HuggingFaceVectorizer = NotInstalled(
        "HuggingFaceVectorizer", "transformers"
    )

try:
    from skembeddings.feature_extraction.text.spacy import SpacyVectorizer
except ModuleNotFoundError:
    SpacyVectorizer = NotInstalled("SpacyVectorizer", "spacy")

__all__ = ["HuggingFaceVectorizer", "SpacyVectorizer"]
