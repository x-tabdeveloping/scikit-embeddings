from typing import Iterable, Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
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
        res = []
        for text in X:
            res.append(self.tokenizer.tokenize(text))
        return res


class HuggingFaceVectorizer(CountVectorizer):
    def __init__(
        self,
        model: str = "bert-base-uncased",
        ngram_range: tuple[int, int] = (1, 1),
        max_df: float = 1.0,
        min_df=1,
        max_features: Optional[int] = None,
    ):
        self.tokenizer = HuggingFaceTokenizer(model)
        super(HuggingFaceVectorizer, self).__init__(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            analyzer=self.tokenizer.transform,  # type: ignore
        )
