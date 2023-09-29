from typing import Literal

from confection import registry

from skembeddings.error import NotInstalled
from skembeddings.models.doc2vec import ParagraphEmbedding
from skembeddings.models.word2vec import Word2VecEmbedding

try:
    from skembeddings.models.glove import GloVeEmbedding
except ModuleNotFoundError:
    ParagraphEmbedding = NotInstalled("GloVeEmbedding", "glove")


@registry.models.register("glove_embedding.v1")
def make_glove_embedding(
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
    return GloVeEmbedding(
        n_components=n_components,
        agg=agg,
        alpha=alpha,
        window=window,
        symmetric=symmetric,
        distance_weighting=distance_weighting,
        iter=iter,
        initial_learning_rate=initial_learning_rate,
        n_jobs=n_jobs,
        memory=memory,
    )


@registry.models.register("word2vec_embedding.v1")
def make_word2vec_embedding(
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
    return Word2VecEmbedding(
        n_components=n_components,
        window=window,
        algorithm=algorithm,
        agg=agg,
        epochs=epochs,
        random_state=random_state,
        negative=negative,
        ns_exponent=ns_exponent,
        cbow_agg=cbow_agg,
        sample=sample,
        hs=hs,
        batch_words=batch_words,
        shrink_windows=shrink_windows,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        n_jobs=n_jobs,
    )


@registry.models.register("paragraph_embedding.v1")
def make_paragraph_embedding(
    n_components: int = 100,
    window: int = 5,
    algorithm: Literal["dm", "dbow"] = "dm",
    tagging_scheme: Literal["hash", "closest"] = "hash",
    max_docs: int = 100_000,
    epochs: int = 10,
    random_state: int = 0,
    negative: int = 5,
    ns_exponent: float = 0.75,
    dm_agg: Literal["mean", "sum", "concat"] = "mean",
    dm_tag_count: int = 1,
    dbow_words: bool = False,
    sample: float = 0.001,
    hs: bool = False,
    batch_words: int = 10000,
    shrink_windows: bool = True,
    learning_rate: float = 0.025,
    min_learning_rate: float = 0.0001,
    n_jobs: int = 1,
):
    return ParagraphEmbedding(
        n_components=n_components,
        window=window,
        algorithm=algorithm,
        tagging_scheme=tagging_scheme,
        max_docs=max_docs,
        epochs=epochs,
        random_state=random_state,
        negative=negative,
        ns_exponent=ns_exponent,
        dm_agg=dm_agg,
        dm_tag_count=dm_tag_count,
        dbow_words=dbow_words,
        sample=sample,
        hs=hs,
        batch_words=batch_words,
        shrink_windows=shrink_windows,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        n_jobs=n_jobs,
    )


__all__ = ["Word2VecEmbedding", "ParagraphEmbedding"]
