import tempfile
from typing import Iterable, Literal

import numpy as np
from confection import Config, registry
from gensim.models import KeyedVectors
from glovpy import GloVe
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from skembeddings.base import Serializable


class GloVeEmbedding(BaseEstimator, TransformerMixin, Serializable):
    def __init__(
        self,
        n_components: int = 100,
        agg: Literal["mean", "max", "both"] = "mean",
        alpha: float = 0.75,
        window: int = 15,
        symmetric: bool = True,
        distance_weighting: bool = True,
        iter: int = 25,
        initial_learning_rate: float = 0.05,
        n_jobs: int = 8,
        memory: float = 4.0,
    ):
        self.agg = agg
        self.n_components = n_components
        self.alpha = alpha
        self.window = window
        self.symmetric = symmetric
        self.distance_weighting = distance_weighting
        self.iter = iter
        self.initial_learning_rate = initial_learning_rate
        self.n_jobs = n_jobs
        self.memory = memory
        self.model_ = None
        self.loss_: list[float] = []
        self.n_features_out = (
            self.n_components if agg != "both" else self.n_components * 2
        )

    def fit(self, X: Iterable[list[str]], y=None):
        self.model_ = GloVe(
            vector_size=self.n_components,
            alpha=self.alpha,
            window_size=self.window,
            symmetric=self.symmetric,
            distance_weighting=self.distance_weighting,
            iter=self.iter,
            initial_learning_rate=self.initial_learning_rate,
            threads=self.n_jobs,
            memory=self.memory,
        )
        self.model_.train(X)
        return self

    def _collect_vectors_single(self, tokens: list[str]) -> np.ndarray:
        embeddings = []
        for token in tokens:
            try:
                embeddings.append(self.model_.wv[token])  # type: ignore
            except KeyError:
                continue
        if not embeddings:
            return np.full((1, self.n_features_out), np.nan)
        return np.stack(embeddings)

    def transform(self, X: Iterable[list[str]], y=None):
        """Transforms the phrase text into a numeric
        representation using word embeddings."""
        if self.model_ is None:
            raise NotFittedError("Model has not been fitted yet.")
        embeddings = []
        for doc in X:
            if not len(doc):
                embeddings.append(np.full(self.n_features_out, np.nan))
                continue
            doc_vectors = self._collect_vectors_single(doc)
            if self.agg == "mean":
                embeddings.append(np.nanmean(doc_vectors, axis=0))
            elif self.agg == "max":
                embeddings.append(np.nanmax(doc_vectors, axis=0))
            elif self.agg == "both":
                mean_vector = np.nanmean(doc_vectors, axis=0)
                max_vector = np.nanmax(doc_vectors, axis=0)
                embeddings.append(np.concatenate((mean_vector, max_vector)))
        return np.stack(embeddings)

    def to_bytes(self) -> bytes:
        if self.model_ is None:
            raise NotFittedError(
                "Can't save model if it hasn't been fitted yet."
            )
        with tempfile.NamedTemporaryFile(prefix="glove-model-") as tmp:
            temporary_filepath = tmp.name
            self.model_.wv.save(temporary_filepath)
            with open(temporary_filepath, "rb") as temp_buffer:
                return temp_buffer.read()

    def from_bytes(self, data: bytes):
        with tempfile.NamedTemporaryFile(prefix="glove-model-") as tmp:
            tmp.write(data)
            keyed_vectors = KeyedVectors.load(tmp.name)
            self.model_ = GloVe()
            self.model_.wv = keyed_vectors
        return self

    @property
    def config(self) -> Config:
        return Config(
            {
                "embedding": {
                    "@models": "glove_embedding.v1",
                    **self.get_params(),
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "GloVeEmbedding":
        resolved = registry.resolve(config)
        return resolved["embedding"]
