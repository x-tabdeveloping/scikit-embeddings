from skembeddings.error import NotInstalled

try:
    from skembeddings.models.word2vec import Word2VecEmbedding
except ModuleNotFoundError:
    Word2VecEmbedding = NotInstalled("Word2VecEmbedding", "gensim")

try:
    from skembeddings.models.doc2vec import Doc2VecEmbedding
except ModuleNotFoundError:
    Doc2VecEmbedding = NotInstalled("Doc2VecEmbedding", "gensim")

__all__ = ["Word2VecEmbedding", "Doc2VecEmbedding"]
