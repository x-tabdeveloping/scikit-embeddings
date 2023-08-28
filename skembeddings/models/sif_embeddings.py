from collections import Counter
from pathlib import Path
from typing import Iterable, Literal, Union

import numpy as np
from gensim import downloader
from gensim.models import KeyedVectors
from gensim.utils import SaveLoad
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

from skembeddings.streams.utils import deeplist


class SIFEmbedding(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        embedding_model: TransformerMixin,
        smoothing: float = 1.0,
    ):
        self.embedding_model = embedding_model
        self.smoothing = smoothing
        self.sing_vect_ = None

    def fit_transform(self, X: Iterable[Iterable[str]], y=None) -> np.ndarray:
        X_eval = deeplist(X)
        X_trf = self.embedding_model.fit_transform(X)
        self.n_features_out = X_trf.shape[1]
        self.freq_ = Counter()
        for sent in X_eval:
            self.freq_.update(sent)
        sent_embeddings = []
        self.total_ = self.freq_.total()
        for sent in X_eval:
            sent_embeddings.append(self._get_embedding(sent))
        sent_embeddings = np.stack(sent_embeddings)
        sent_embeddings = SimpleImputer().fit_transform(sent_embeddings)
        self.svd_ = TruncatedSVD(1).fit(sent_embeddings)
        self.sing_vect_ = self.svd_.components_
        u = self.sing_vect_
        X_new = np.stack([sent - u.T @ u @ sent for sent in sent_embeddings])
        return X_new

    def _get_embedding(self, sent: list[str]) -> np.ndarray:
        if not sent:
            return np.full(self.n_features_out, np.nan)
        embeddings = self.embedding_model.transform(sent)
        embeddings = np.stack(embeddings)
        freqs = np.array([self.freq_[token] / self.total_ for token in sent])
        weighted: np.ndarray = (
            embeddings.T * (self.smoothing / (self.smoothing + freqs))
        ).T
        return np.sum(weighted, axis=0) / len(sent)

    def fit(self, X: Iterable[Iterable[str]], y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X: Iterable[Iterable[str]]) -> np.ndarray:
        if self.sing_vect_ is None:
            raise NotFittedError("Please fit the model before transforming.")
        X_eval: list[list[str]] = deeplist(X)
        sent_embeddings = []
        for sent in X_eval:
            sent_embeddings.append(self._get_embedding(sent))
        sent_embeddings = np.stack(sent_embeddings)
        u = self.sing_vect_
        X_new = np.stack([sent - u.T @ u @ sent for sent in sent_embeddings])
        return X_new
