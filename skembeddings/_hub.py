DEFAULT_README = """
---
language:
    - en
tags:
    - embeddings
    - tokenizers
library_name: scikit-embeddings
---

# {repo}

This repository contains an embedding pipeline that has been trained using [scikit-embeddings](https://github.com/x-tabdeveloping/scikit-embeddings)

## Usage
```python
# pip install scikit-embeddings

from skembeddings.pipeline import EmbeddingPipeline

pipe = EmbeddingPipeline.from_hub('{repo}')

pipe.transform(["A text you intend to vectorize."])
```
"""
