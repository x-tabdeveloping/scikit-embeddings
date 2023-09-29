import tempfile
from typing import Iterable, Literal

import numpy as np
from confection import Config, registry
from gensim.models import KeyedVectors, Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from skembeddings.base import Serializable


class Word2VecEmbedding(BaseEstimator, TransformerMixin, Serializable):
    """Scikit-learn compatible Word2Vec embedding component.

    Parameters
    ----------
    n_components: int, default 100
        Desired size of the embeddings.
    window: int, default 5
        Window size over tokens.
    algorithm: Literal["cbow", "sg"], default "cbow"
        Indicates whether a continuous bag-of-words or
        skip-gram model should be trained.
    agg: Literal["mean", "max", "both"], default "mean"
        Indicates how output words should be aggregated when inferring
        embeddings for input documents.
    epochs: int, default 5
        Number of training epochs to run.
    random_state: int, default 0
        Random seed for reproducible results.
    negative: int, default 5
        If > 0, negative sampling will be used,
        the int for negative specifies how many “noise words”
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    ns_exponent: float, default 0.75
        The exponent used to shape the negative sampling distribution.
        A value of 1.0 samples exactly in proportion to the frequencies,
        0.0 samples all words equally, while a negative value samples
        low-frequency words more than high-frequency words.
        The popular default value of 0.75
        was chosen by the original Word2Vec paper.
    cbow_agg: Literal["mean", "sum"], default "mean"
        Indicates how context words should be aggregated when using
        CBOW.
    sample: float, default 0.001
        The threshold for configuring which higher-frequency words are
        randomly downsampled, useful range is (0, 1e-5).
    hs: bool, default False
        Indicates whether hierarchical softmax should be used.
        If set to False and negative is nonzero, negative sampling will
        be used as the training objective.
    batch_words: int, default 10000
        Target size (in words) for batches of
        examples passed to worker threads.
    shrink_windows: bool, default True
        When True, the effective window size is randomly sampled
        from [1, window].
    learning_rate: float, default 0.025
        Learning rate for the optimizer.
    min_learning_rate: float, default 0.0001
        The learning rate will linearly drop to this
        value during training.
    n_jobs: int, default 1
        Number of cores to use for training.
    """

    def __init__(
        self,
        n_components: int = 100,
        window: int = 5,
        algorithm: Literal["cbow", "sg"] = "cbow",
        agg: Literal["mean", "max", "both"] = "mean",
        epochs: int = 5,
        random_state: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        cbow_agg: Literal["mean", "sum"] = "mean",
        sample: float = 0.001,
        hs: bool = False,
        batch_words: int = 10000,
        shrink_windows: bool = True,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        n_jobs: int = 1,
    ):
        self.agg = agg
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.window = window
        self.algorithm = algorithm
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_agg = cbow_agg
        self.sample = sample
        self.hs = hs
        self.batch_words = batch_words
        self.shrink_windows = shrink_windows
        self.epochs = epochs
        self.model_ = None
        self.n_features_out = (
            self.n_components if agg != "both" else self.n_components * 2
        )
        agg_options = ["mean", "max", "both"]
        if self.agg not in agg_options:
            raise ValueError(
                f"The `agg` value must be in {agg_options}. Got {self.agg}."
            )

    def _init_model(self, sentences=None) -> Word2Vec:
        return Word2Vec(
            sentences=sentences,
            vector_size=self.n_components,
            min_count=0,
            alpha=self.learning_rate,
            window=self.window,
            sample=self.sample,
            seed=self.random_state,
            workers=self.n_jobs,
            min_alpha=self.min_learning_rate,
            sg=int(self.algorithm == "sg"),
            hs=int(self.hs),
            negative=self.negative,
            ns_exponent=self.ns_exponent,
            cbow_mean=int(self.cbow_agg == "mean"),
            epochs=self.epochs,
            trim_rule=None,
            batch_words=self.batch_words,
            compute_loss=True,
            shrink_windows=self.shrink_windows,
        )

    def fit(self, X: Iterable[list[str]], y=None):
        self.loss_ = []
        self.model_ = self._init_model(sentences=X)
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

    @property
    def keyed_vectors(self) -> KeyedVectors:
        if self.model_ is None:
            raise NotFittedError(
                "Can't access keyed vectors, model has not been fitted yet."
            )
        return self.model_.wv

    def to_bytes(self) -> bytes:
        if self.model_ is None:
            raise NotFittedError(
                "Can't save model if it hasn't been fitted yet."
            )
        with tempfile.NamedTemporaryFile(prefix="gensim-model-") as tmp:
            temporary_filepath = tmp.name
            self.model_.save(temporary_filepath)
            with open(temporary_filepath, "rb") as temp_buffer:
                return temp_buffer.read()

    def from_bytes(self, data: bytes) -> "Word2VecEmbedding":
        with tempfile.NamedTemporaryFile(prefix="gensim-model-") as tmp:
            tmp.write(data)
            model = Word2Vec.load(tmp.name)
            self.model_ = model
        return self

    @property
    def config(self) -> Config:
        return Config(
            {
                "embedding": {
                    "@models": "word2vec_embedding.v1",
                    **self.get_params(),
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "Word2VecEmbedding":
        resolved = registry.resolve(config)
        return resolved["embedding"]
