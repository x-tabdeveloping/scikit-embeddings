from skembeddings.error import NotInstalled

try:
    from skembeddings.models.word2vec import Word2VecEmbedding
except ModuleNotFoundError:
    Word2VecEmbedding = NotInstalled("Word2VecEmbedding", "gensim")

try:
    from skembeddings.models.doc2vec import ParagraphEmbedding
except ModuleNotFoundError:
    ParagraphEmbedding = NotInstalled("ParagraphEmbedding", "gensim")

try:
    from skembeddings.models.sif_embeddings import SIFEmbedding
except ModuleNotFoundError:
    SIFEmbedding = NotInstalled("SIFEmbedding", "gensim")

__all__ = ["Word2VecEmbedding", "ParagraphEmbedding", "SIFEmbedding"]
