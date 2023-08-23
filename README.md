# scikit-embeddings
Utilites for training word and document embeddings in scikit-learn pipelines.

This is a module that is currently in development and experimental phase, DO NOT use it for production code.

## Features
 - Stream texts easily from disk and chunk them so you can use large datasets for training embeddings.
 - HuggingFace and pure Python tokenizers for highest performance and accuracy for downstream tasks.
 - spaCy tokenizers with lemmatization, stop word removal and augmentation with POS-tags/Morphological information etc. for highest quality embeddings for literary analysis.
 - Easy to integrate components and pipelines in your scikit-learn workflows and machine learning pipelines.
 - Word2Vec and Doc2Vec components with more to come in the future.

## Example Usage

Here's an example of how you can create a fast word embedding pipeline for highest downstream performance.

```python
from skpartial.pipeline import make_partial_pipeline

from skembeddings.models.word2vec import Word2VecEmbedding
from skembeddings.tokenization import HuggingFaceTokenizer
from skembedding.streams import Stream

# let's say you have a list of file paths
files: list[str] = [...]

# Stream text chunks from jsonl files with a 'content' field.
text_chunks = (
  Stream(files)
  .read_files(lines=True)
  .json()
  .grab("content")
  .chunk(10_000)
)

# Build a pipeline
embedding_pipeline = make_partial_pipeline(
    HuggingFaceTokenizer("bert-base-uncased"),
    Word2VecEmbedding(100, algorithm="cbow")
)

for batch in text_chunks:
    embedding_pipeline.partial_fit(batch)

```

And an example of how you could build a Sense2Vec model on a small carefully curated corpus:

```python
from sklearn.pipeline import make_pipeline

from skembeddings.models.word2vec import Word2VecEmbedding
from skembeddings.tokenization import SpacyTokenizer
from skembedding.streams import Stream

corpus: list[str] = [...]

# spaCy tokenizer that lemmatizes, removes stop words and augments tokens
# with POS-tags.

# Single token pattern that lets alphabetical tokens pass, but not stopwords
pattern = [[{"IS_ALPHA": True, "IS_STOP": False}]]
tokenizer = SpacyTokenizer(
    "en_core_web_sm",
    out_attrs=("LEMMA", "UPOS"),
    patterns=pattern,
)

# Build a pipeline
embedding_pipeline = make_pipeline(
    tokenizer,
    Word2VecEmbedding(50, algorithm="sg", epochs=5)
)

# Fitting pipeline on corpus
embedding_pipeline.fit(corpus)
```
