<img align="left" width="82" height="82" src="assets/logo.svg">

# scikit-embeddings

<br>
Utilites for training, storing and using word and document embeddings in scikit-learn pipelines.

## Features
 - Train Word and Paragraph embeddings in scikit-learn compatible pipelines.
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

If you want to use GloVe embedding models, install alogn with glovpy:

```bash
pip install scikit-embeddings[glove]
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

