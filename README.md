# scikit-embeddings
Utilites for training word, document and sentence embeddings in scikit-learn pipelines.

## Features
 - Train Word, Paragraph or Sentence embeddings in scikit-learn compatible pipelines.
 - Stream texts easily from disk and chunk them so you can use large datasets for training embeddings.
 - spaCy tokenizers with lemmatization, stop word removal and augmentation with POS-tags/Morphological information etc. for highest quality embeddings for literary analysis.
 - Fast and performant trainable tokenizer components from `tokenizers`.
 - Easy to integrate components and pipelines in your scikit-learn workflows and machine learning pipelines.
 - Easy serialization and integration with HugginFace Hub for quickly publishing your embedding pipelines.

### What scikit-embeddings is not for:
 - Training transformer models and deep neural language models (if you want to do this, do it with [transformers](https://huggingface.co/docs/transformers/index))
 - Using pretrained sentence transformers (use [embetter](https://github.com/koaning/embetter))

## Installation

You can easily install scikit-embeddings from PyPI:

```bash
pip install scikit-embeddings
```

If you want to use Gensim embedding models, install alogn with gensim:

```bash
pip install scikit-embeddings[gensim]
```

If you intend to use spacy tokenizers, install spacy:

```bash
pip install scikit-embeddings[spacy]
```

## Streams

scikit-embeddings comes with a handful of utilities for streaming data from disk or other sources,
chunking and filtering. Here's an example of how you would go about obtaining chunks of text from jsonl files with a "content field".

```python
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
```

## Example Pipelines

You can use scikit-embeddings with many many different pipeline architectures, I will list a few here:

### Word Embeddings

You can train classic vanilla word embeddings by building a pipeline that contains a `WordLevel` tokenizer and an embedding model:

```python
from skembedding.tokenizers import WordLevelTokenizer
from skembedding.models import Word2VecEmbedding
from skembeddings.pipeline import EmbeddingPipeline

embedding_pipe = EmbeddingPipeline(
    WordLevelTokenizer(),
    Word2VecEmbedding(n_components=100, algorithm="cbow")
)
embedding_pipe.fit(texts)
```

### Fasttext-like

You can train an embedding pipeline that uses subword information by using a tokenizer that does that.
You may want to use `Unigram`, `BPE` or `WordPiece` for these purposes.
Fasttext also uses skip-gram by default so let's change to that.

```python
from skembedding.tokenizers import UnigramTokenizer
from skembedding.models import Word2VecEmbedding
from skembeddings.pipeline import EmbeddingPipeline

embedding_pipe = EmbeddingPipeline(
    UnigramTokenizer(),
    Word2VecEmbedding(n_components=250, algorithm="sg")
)
embedding_pipe.fit(texts)
```

### Sense2Vec

We provide a spaCy tokenizer that can lemmatize tokens and append morphological information so you can get fine-grained
semantic information even on relatively small corpora. I recommend using this for literary analysis.

```python
from skembeddings.models import Word2VecEmbedding
from skembeddings.tokenizers import SpacyTokenizer
from skembeddings.pipeline import EmbeddingPipeline

# Single token pattern that lets alphabetical tokens pass, but not stopwords
pattern = [[{"IS_ALPHA": True, "IS_STOP": False}]]

# Build tokenizer that lemmatizes and appends POS-tags to the lemmas
tokenizer = SpacyTokenizer(
    "en_core_web_sm",
    out_attrs=("LEMMA", "UPOS"),
    patterns=pattern,
)

# Build a pipeline
embedding_pipeline = EmbeddingPipeline(
    tokenizer,
    Word2VecEmbedding(50, algorithm="cbow")
)

# Fitting pipeline on corpus
embedding_pipeline.fit(corpus)
```

### Paragraph Embeddings

You can train Doc2Vec paragpraph embeddings with the chosen choice of tokenization.

```python
from skembedding.tokenizers import WordPieceTokenizer
from skembedding.models import ParagraphEmbedding
from skembeddings.pipeline import EmbeddingPipeline

embedding_pipe = EmbeddingPipeline(
    WordPieceTokenizer(),
    ParagraphEmbedding(n_components=250, algorithm="dm")
)
embedding_pipe.fit(texts)
```

## Iterative training

In the case of large datasets you can train on individual chunks with `partial_fit()`.

```python
for chunk in text_chunks:
    embedding_pipe.partial_fit(chunk)
```

## Serialization

Pipelines can be safely serialized to disk:

```python
embedding_pipe.to_disk("output_folder/")

embedding_pipe = EmbeddingPipeline.from_disk("output_folder/")
```

Or published to HugginFace Hub:

```python
from huggingface_hub import login

login()
embedding_pipe.to_hub("username/name_of_pipeline")

embedding_pipe = EmbeddingPipeline.from_hub("username/name_of_pipeline")
```

## Text Classification

You can include an embedding model in your classification pipelines by adding some classification head.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y)

cls_pipe = make_pipeline(embedding_pipe, LogisticRegression())
cls_pipe.fit(X_train, y_train)

y_pred = cls_pipe.predict(X_test)
print(classification_report(y_test, y_pred))
```


## Feature Extraction

If you intend to use the features produced by tokenizers in other text pipelines, such as topic models,
you can use `ListCountVectorizer` or `Joiner`.

Here's an example of an NMF topic model that use lemmata enriched with POS tags.

```python
from sklearn.decomposition import NMF
from sklearn.pipelines import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from skembedding.tokenizers import SpacyTokenizer
from skembedding.feature_extraction import ListCountVectorizer
from skembedding.preprocessing import Joiner

# Single token pattern that lets alphabetical tokens pass, but not stopwords
pattern = [[{"IS_ALPHA": True, "IS_STOP": False}]]

# Build tokenizer that lemmatizes and appends POS-tags to the lemmas
tokenizer = SpacyTokenizer(
    "en_core_web_sm",
    out_attrs=("LEMMA", "UPOS"),
    patterns=pattern,
)

# Example with ListCountVectorizer
topic_pipeline = make_pipeline(
    tokenizer,
    ListCountVectorizer(),
    TfidfTransformer(), # tf-idf weighting (optional)
    NMF(15), # 15 topics in the model 
)

# Alternatively you can just join the tokens together with whitespace
topic_pipeline = make_pipeline(
    tokenizer,
    Joiner(),
    TfidfVectorizer(),
    NMF(15), 
)
```
