from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer


class HuggingFaceTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, model: str = "bert-base-uncased"):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

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
            res.append(self.tokenizer.tokenize(text))
        return res

    def get_feature_names_out(self, input_features=None):
        return None
