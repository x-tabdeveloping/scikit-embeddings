import tempfile
from typing import Iterable, Literal

import numpy as np
from confection import Config, registry
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import murmurhash3_32

from skembeddings.base import Serializable


def deeplist(nested) -> list:
    """Recursively turns nested iterable to list.

    Parameters
    ----------
    nested: iterable
        Nested iterable.

    Returns
    -------
    list
        Nested list.
    """
    if not isinstance(nested, Iterable) or isinstance(nested, str):
        return nested  # type: ignore
    else:
        return [deeplist(sub) for sub in nested]


def _tag_enumerate(docs: Iterable[list[str]]) -> list[TaggedDocument]:
    """Tags documents with their integer positions."""
    return [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]


class ParagraphEmbedding(BaseEstimator, TransformerMixin, Serializable):
    """Scikit-learn compatible Paragraph Embedding model.

    Parameters
    ----------
    n_components: int, default 100
        Desired size of the embeddings.
    window: int, default 5
        Window size over tokens.
    algorithm: Literal["dm", "dbow"], default "dm"
        Indicates whether distributed memory or distributed
        bag of words model gets trained.
    max_docs: int, default 100_000
        Number of maximum documents to keep the embeddings of.
        Tags for further documents get resolved with the specified tagging
        scheme.
    tagging_scheme: Literal["hash", "closest"], default "hash"
        Specifies what tags should be assigned to new documents after
        the number of maximum slots has been filled. "hash" hashes the document
        and tags it with a hash mod index. "closest" assigns the tag of the
        closest document in the model's stored embeddings.
    random_state: int, default 0
        Random seed so that training is reproducible.
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
    dm_agg: Literal["mean", "sum", "concat"], default "mean"
        Specifies how context vectors should be aggregated.
    dm_tag_count: int, default 1
        Expected constant number of document tags per document,
        when using dm_agg=='concat'.
    dbow_words: bool, default False
        When True trains word-vectors (in skip-gram fashion) simultaneous with
        DBOW doc-vector training; When False, only trains doc-vectors (faster).
    sample: float, default 0.001
        The threshold for configuring which higher-frequency words are
        randomly downsampled, useful range is (0, 1e-5).
    hs: bool, default False
        Indicates whether hierarchical softmax should be used.
        If set to False and negative is nonzero, negative sampling will
        be used as the training objective.
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

    Attributes
    ----------
    model_: Doc2Vec
        Underlying gensim doc2vec model.
    loss_: list[float]
        Loss history during training.
    seen_docs_: int
        Number of seen documents.
    config: Config
        Generated config object for model serialization.
    """

    def __init__(
        self,
        n_components: int = 100,
        window: int = 5,
        algorithm: Literal["dm", "dbow"] = "dm",
        max_docs: int = 100_000,
        tagging_scheme: Literal["hash", "closest"] = "hash",
        random_state: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        dm_agg: Literal["mean", "sum", "concat"] = "mean",
        dm_tag_count: int = 1,
        dbow_words: bool = False,
        sample: float = 0.001,
        hs: bool = False,
        shrink_windows: bool = True,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        n_jobs: int = 1,
    ):
        self.model_ = None
        self.loss_: list[float] = []
        self.seen_docs_ = 0
        if tagging_scheme not in ["hash", "closest"]:
            raise ValueError(
                "Tagging scheme should either be 'hash' or 'closest'"
            )
        self.algorithm = algorithm
        self.max_docs = max_docs
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.window = window
        self.tagging_scheme = tagging_scheme
        self.random_state = random_state
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.dm_agg = dm_agg
        self.dm_tag_count = dm_tag_count
        self.dbow_words = dbow_words
        self.sample = sample
        self.hs = hs
        self.shrink_windows = shrink_windows
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate

    def _tag_documents(
        self, documents: list[list[str]]
    ) -> list[TaggedDocument]:
        if self.model_ is None:
            raise TypeError(
                "You should not call _tag_documents"
                "before model is initialised."
            )
        res = []
        for document in documents:
            # While we have available slots we just add new documents to those
            if self.seen_docs_ < self.max_docs:
                res.append(TaggedDocument(document, [self.seen_docs_]))
            else:
                # If we run out, we choose a tag based on a scheme
                if self.tagging_scheme == "hash":
                    # Here we use murmur hash
                    hash = murmurhash3_32("".join(document))
                    id = hash % self.max_docs
                    res.append(TaggedDocument(document, [id]))
                elif self.tagging_scheme == "closest":
                    # We obtain the key of the most semantically
                    # similar document and use that.
                    doc_vector = self.model_.infer_vector(document)
                    key, _ = self.model_.dv.similar_by_key(doc_vector, topn=1)[
                        0
                    ]
                    res.append(TaggedDocument(document, [key]))
                else:
                    raise ValueError(
                        "Tagging scheme should either be 'hash' or 'closest'"
                        f" but {self.tagging_scheme} was provided."
                    )
            self.seen_docs_ += 1
        return res

    def _init_model(self, docs=None) -> Doc2Vec:
        return Doc2Vec(
            documents=docs,
            vector_size=self.n_components,
            min_count=0,
            alpha=self.learning_rate,
            window=self.window,
            sample=self.sample,
            seed=self.random_state,
            workers=self.n_jobs,
            min_alpha=self.min_learning_rate,
            dm=int(self.algorithm == "dm"),
            dm_mean=int(self.dm_agg == "mean"),
            dm_concat=int(self.dm_agg == "concat"),
            dbow_words=int(self.dbow_words),
            dm_tag_count=self.dm_tag_count,
            hs=int(self.hs),
            negative=self.negative,
            ns_exponent=self.ns_exponent,
            epochs=1,
            trim_rule=None,
            compute_loss=True,
            shrink_windows=self.shrink_windows,
        )

    def _append_loss(self):
        self.loss_.append(self.model_.get_latest_training_loss())  # type: ignore

    def fit(self, X: Iterable[Iterable[str]], y=None):
        """Fits a new paragraph embedding model to the given documents.

        Parameters
        ----------
        X: Iterable[Iterable[str]]
            List of documents as list of tokens.
        y: None
            Ignored.

        Returns
        -------
        Self
            Fitted model.
        """
        self.seen_docs_ = 0
        # Forcing evaluation
        X_eval: list[list[str]] = deeplist(X)
        n_docs = len(X_eval)
        if self.max_docs < n_docs:
            init_batch = _tag_enumerate(X_eval[: self.max_docs])
            self.model_ = self._init_model(init_batch)
            self._append_loss()
            self.partial_fit(X_eval[self.max_docs :])
            return self
        docs = _tag_enumerate(X_eval)
        self.model_ = self._init_model(docs)
        self._append_loss()
        return self

    def partial_fit(self, X: Iterable[Iterable[str]], y=None):
        """Partially fits doc2vec model (online fitting).

        Parameters
        ----------
        X: Iterable[Iterable[str]]
            List of documents as list of tokens.
        y: None
            Ignored.

        Returns
        -------
        Self
            Fitted model.
        """
        # Force evaluation on iterable
        X_eval: list[list[str]] = deeplist(X)
        if self.model_ is None:
            self.fit(X_eval)
            return self
        # We obtained tagged documents
        tagged_docs = self._tag_documents(X_eval)
        # Then build vocabulary
        self.model_.build_vocab(tagged_docs, update=True)
        self.model_.train(
            tagged_docs,
            total_examples=self.model_.corpus_count,
            epochs=1,
            compute_loss=True,
        )
        self._append_loss()
        return self

    def transform(self, X: Iterable[Iterable[str]]) -> np.ndarray:
        """Infers vectors for all of the given documents.

        Parameters
        ----------
        X: Iterable[Iterable[str]]
            List of documents as list of tokens.

        Returns
        -------
        ndarray of shape (n_docs, n_components)
            Inferred document embeddings.
        """
        if self.model_ is None:
            raise NotFittedError(
                "Model ha been not fitted, please fit before inference."
            )
        vectors = [self.model_.infer_vector(list(doc)) for doc in X]
        return np.stack(vectors)

    def to_bytes(self) -> bytes:
        """Serializes model to a bytes object."""
        if self.model_ is None:
            raise NotFittedError(
                "Can't save model if it hasn't been fitted yet."
            )
        with tempfile.NamedTemporaryFile(prefix="gensim-model-") as tmp:
            temporary_filepath = tmp.name
            self.model_.save(temporary_filepath)
            with open(temporary_filepath, "rb") as temp_buffer:
                return temp_buffer.read()

    def from_bytes(self, data: bytes) -> "ParagraphEmbedding":
        """Loads and assigns serialized model from bytes."""
        with tempfile.NamedTemporaryFile(prefix="gensim-model-") as tmp:
            tmp.write(data)
            model = Doc2Vec.load(tmp.name)
            self.model_ = model
        return self

    @property
    def config(self) -> Config:
        return Config(
            {
                "embedding": {
                    "@models": "paragraph_embedding.v1",
                    **self.get_params(),
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "ParagraphEmbedding":
        """Initialize model from config object."""
        resolved = registry.resolve(config)
        return resolved["embedding"]
