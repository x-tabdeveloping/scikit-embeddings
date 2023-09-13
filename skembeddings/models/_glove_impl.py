from collections import Counter
from typing import Iterable, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numba import njit
from tqdm import tqdm


def vectorize_ragged(
    docs: list[list[str]], vocab: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.zeros(len(docs), dtype=np.uint32)
    tokens = []
    for i_doc, doc in enumerate(docs):
        lengths[i_doc] = len(doc)
        for token in doc:
            tokens.append(vocab[token])
    return np.array(tokens, dtype=np.uint32), lengths


@njit(fastmath=True)
def count_cooccurrences(
    tokens: np.ndarray, lengths: np.ndarray, context_size: int
) -> dict[tuple[int, int], float]:
    cooccurrences = {(0, 0): 0.1}
    offsets = np.cumsum(lengths)
    i_doc = 0
    i_context_start = 0
    i_context_end = i_context_start + context_size * 2 + 1
    i_target = i_context_start + context_size + 1
    while i_context_end < len(tokens):
        while i_context_end < offsets[i_doc]:
            for i_context in range(i_context_start, i_context_end):
                if i_target != i_context:
                    target, context = tokens[i_target], tokens[i_context]
                    distance = abs(i_target - i_context)
                    if (target, context) in cooccurrences:
                        cooccurrences[(target, context)] += 1 / distance
                    else:
                        cooccurrences[(target, context)] = 1 / distance
            i_context_start += 1
            i_context_end += 1
            i_target += 1
        i_context_start = offsets[i_doc]
        i_context_end = i_context_start + context_size * 2 + 1
        i_target = i_context_start + context_size + 1
        i_doc += 1
    return cooccurrences


def forward(
    target,
    context,
    coocc,
    embeddings,
    context_embeddings,
    bias,
    context_bias,
    max_coocc,
    alpha=0.75,
):
    w_embedding = jnp.take(embeddings, target, axis=0)
    w_context = jnp.take(context_embeddings, context, axis=0)
    b_target = jnp.take(bias, target, axis=0)
    b_context = jnp.take(context_bias, context, axis=0)
    loss = jnp.matmul(w_embedding, w_context.T).sum(axis=1)
    loss = jnp.square(loss + b_target + b_context - jnp.log(coocc))
    weighted_coocc = jnp.clip(jnp.float_power(coocc / max_coocc, alpha), 0, 1)
    loss = jnp.mean(jnp.matmul(weighted_coocc, loss))
    return loss


def loss_fn(
    params,
    target,
    context,
    coocc,
    max_coocc,
    alpha=0.75,
):
    return forward(
        target, context, coocc, max_coocc=max_coocc, alpha=alpha, **params
    )


class GloVeModel:
    def __init__(
        self,
        vector_size: int = 100,
        alpha: float = 0.75,
        seed: int = 0,
        window_size: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-2,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.vector_size = vector_size
        self.window_size = window_size
        self.embeddings = None
        self.context_embeddings = None
        self.bias = None
        self.context_bias = None
        self.vocab = dict({"[UNK]": 0})
        self.next_index = 1
        self.cooccurrences = Counter()
        self.batch_size = batch_size
        self.alpha = alpha
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.loss_history = []

    # def tokens_to_indices(self, docs: list[list[str]]) -> list[np.ndarray]:
    #     res = []
    #     for doc in docs:
    #         idx = [self.vocab[token] for token in doc]
    #         res.append(np.array(idx, dtype=np.uint32))
    #     return res

    def update_cooccurrences(self, docs: list[list[str]]):
        tokens, lengths = vectorize_ragged(docs, self.vocab)
        print(tokens, lengths)
        print("Counting cooccurrences")
        self.cooccurrences.update(
            count_cooccurrences(tokens, lengths, self.window_size)
        )

    def extend_params(self, new_vocab: list[str]):
        if self.embeddings is None:
            self.embeddings = self.rng.normal(
                size=(len(new_vocab) + 1, self.vector_size)
            )
            self.context_embeddings = self.rng.normal(
                size=(len(new_vocab) + 1, self.vector_size)
            )
            self.bias = self.rng.normal(size=(len(new_vocab) + 1,))
            self.context_bias = self.rng.normal(size=(len(new_vocab) + 1,))
        else:
            self.embeddings = np.concatenate(
                (
                    self.embeddings,
                    self.rng.normal(size=(len(new_vocab), self.vector_size)),
                )
            )
            self.context_embeddings = np.concatenate(
                (
                    self.context_embeddings,
                    self.rng.normal(size=(len(new_vocab), self.vector_size)),
                ),
                axis=0,
            )
            self.bias = np.concatenate(
                (self.bias, self.rng.normal(size=len(new_vocab)))
            )
            self.context_bias = np.concatenate(
                (self.context_bias, self.rng.normal(size=len(new_vocab)))
            )

    def update_vocab(self, docs: list[list[str]]):
        new_tokens = set()
        for doc in docs:
            new_tokens |= set(doc)
        novel_tokens = set()
        for token in new_tokens:
            if token not in self.vocab:
                novel_tokens.add(token)
                self.vocab[token] = self.next_index
                self.next_index += 1
        if new_tokens:
            self.extend_params(list(novel_tokens))

    def batches(self) -> Iterable[dict]:
        entries = jnp.array(
            [
                (target, context, coocc)
                for (target, context), coocc in self.cooccurrences.items()
            ]
        )
        for batch_i in range(0, len(entries), self.batch_size):
            batch_entries = entries[batch_i : batch_i + self.batch_size]
            yield dict(
                target=batch_entries[:, 0],
                context=batch_entries[:, 1],
                coocc=batch_entries[:, 2],
            )

    @property
    def params(self) -> dict:
        return dict(
            embeddings=jnp.array(self.embeddings),
            context_embeddings=jnp.array(self.context_embeddings),
            bias=jnp.array(self.bias),
            context_bias=jnp.array(self.context_bias),
        )

    def update_params(self, params):
        self.embeddings = params["embeddings"]
        self.context_embeddings = params["context_embeddings"]
        self.bias = params["bias"]
        self.context_bias = params["context_bias"]

    def update_weights(self):
        _, max_coocc = self.cooccurrences.most_common(1)[0]
        opt_state = self.optimizer.init(self.params)
        params = self.params

        @jax.jit
        def step(params, opt_state, batch):
            loss_value, grads = jax.value_and_grad(loss_fn)(
                params, **batch, alpha=self.alpha, max_coocc=max_coocc
            )
            updates, opt_state = self.optimizer.update(
                grads, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for batch in tqdm(
            self.batches(),
            desc="Training over cooccurrence batches.",
            total=len(self.cooccurrences) // self.batch_size + 1,
        ):
            params, opt_state, loss_value = step(params, opt_state, batch)
            self.loss_history.append(loss_value)
        self.update_params(params)

    def __getitem__(self, key: Union[str, list[str]]):
        if self.embeddings is None:
            raise TypeError("GloVe Model has not been fitted yet")
        keys = [key] if isinstance(key, str) else key
        indices = []
        for k in keys:
            if k in self.vocab:
                indices.append(self.vocab[k])
        indices = [self.vocab.get(_key, self.vocab["[UNK]"]) for _key in keys]
        if not indices:
            return np.full((1, self.vector_size), np.nan)
        return np.array(self.embeddings)[indices]
