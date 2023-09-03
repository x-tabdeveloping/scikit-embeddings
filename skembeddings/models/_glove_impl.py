from collections import Counter
from typing import Iterable, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


@njit
def count_freq(sequence: list[int]) -> dict[int, int]:
    res = dict()
    for elem in sequence:
        if elem in res:
            res[elem] += 1
        else:
            res[elem] = 1
    return res


# @njit(parallel=True, fastmath=True)
# def count_cooccurrences(
#     docs: list[np.ndarray], context_size: int
# ) -> dict[tuple[int, int], int]:
#     cooccurrences = dict()
#     for doc in docs:
#         windows = sliding_window_view(
#             doc, window_shape=min(context_size, len(doc))
#         )
#         for window in windows:
#             n_occ = count_freq(window)
#             for w1 in n_occ:
#                 for w2 in n_occ:
#                     if w1 == w2:
#                         continue
#                     w1, w2 = min(w1, w2), max(w1, w2)
#                     if (w1, w2) in cooccurrences:
#                         cooccurrences[w1, w2] += n_occ[w1] * n_occ[w2]
#                     else:
#                         cooccurrences[w1, w2] = n_occ[w1] * n_occ[w2]
#     return cooccurrences


@njit(fastmath=True)
def count_cooccurrences(
    doc: np.ndarray, context_size: int
) -> dict[tuple[int, int], int]:
    cooccurrences = dict()
    windows = sliding_window_view(
        doc, window_shape=min(context_size, len(doc))
    )
    for window in windows:
        n_occ = count_freq(window)
        for w1 in n_occ:
            for w2 in n_occ:
                if w1 == w2:
                    continue
                w1, w2 = min(w1, w2), max(w1, w2)
                if (w1, w2) in cooccurrences:
                    cooccurrences[w1, w2] += n_occ[w1] * n_occ[w2]
                else:
                    cooccurrences[w1, w2] = n_occ[w1] * n_occ[w2]
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
    loss = jnp.matmul(w_embedding, w_context.T)
    loss = jnp.sum(loss, axis=1)
    loss = loss + b_target + b_context
    loss = jnp.square(loss - jnp.log(coocc))
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
        self.vocab = dict()
        self.next_index = 0
        self.cooccurrences = Counter()
        self.batch_size = batch_size
        self.alpha = alpha
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.loss_history = []

    def tokens_to_indices(self, docs: list[list[str]]) -> list[np.ndarray]:
        res = []
        for doc in docs:
            idx = [self.vocab[token] for token in doc]
            res.append(np.array(idx, dtype=np.uint32))
        return res

    def update_cooccurrences(self, docs: list[list[str]]):
        indices = self.tokens_to_indices(docs)
        for doc in tqdm(indices, desc="Counting cooccurrences in documents."):
            new_coocc = count_cooccurrences(doc, self.window_size)
            self.cooccurrences.update(new_coocc)

    def extend_params(self, new_vocab: list[str]):
        new_vocab = list(new_vocab)
        if self.embeddings is None:
            self.embeddings = self.rng.normal(
                size=(len(new_vocab), self.vector_size)
            )
            self.context_embeddings = self.rng.normal(
                size=(len(new_vocab), self.vector_size)
            )
            self.bias = self.rng.normal(size=(len(new_vocab),))
            self.context_bias = self.rng.normal(size=(len(new_vocab),))
        else:
            self.embeddings = np.concatenate(
                self.embeddings,
                self.rng.normal(size=(len(new_vocab), self.vector_size)),
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
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_index
                self.next_index += 1
        if new_tokens:
            self.extend_params(list(new_tokens))

    def batches(self) -> Iterable[dict]:
        all_keys = np.array(list(self.cooccurrences))
        self.rng.shuffle(all_keys)
        for batch_i in range(0, len(all_keys), self.batch_size):
            keys = all_keys[batch_i : batch_i + self.batch_size]
            coocc = [self.cooccurrences[tuple(key)] for key in keys]
            flip = bool(self.rng.integers(2))
            if flip:
                keys = np.flip(keys, axis=1)
            keys = jnp.array(keys)
            yield dict(
                target=keys[:, 0], context=keys[:, 1], coocc=jnp.array(coocc)
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
        indices = [self.vocab[_key] for _key in keys]
        if not indices:
            return np.full((1, self.vector_size), np.nan)
        return np.array(self.embeddings)[indices]
