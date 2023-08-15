from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin


class Joiner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sep: str = " ",
    ):
        self.sep = sep

    def fit(self, X, y=None):
        """Exists for compatiblity, doesn't do anything."""
        return self

    def partial_fit(self, X, y=None):
        """Exists for compatiblity, doesn't do anything."""
        return self

    def transform(self, X: Iterable[list[str]]) -> Iterable[str]:
        for doc in X:
            yield self.sep.join(doc)
