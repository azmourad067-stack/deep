"""
Reseau de neurones multicouche (embeddings + MLP) implemente en NumPy pur :
  - Une couche d'embedding par variable categorielle (hippodrome, discipline, sexe)
  - Concatenation avec les features numeriques
  - 2 couches cachees (ReLU) + dropout
  - Deux tetes de sortie partageant le tronc commun :
      * tete "victoire"  -> score par cheval, softmax NORMALISE PAR COURSE
                             (un seul gagnant possible par course = loss
                             cross-entropy multinomiale, pas une classification
                             binaire independante par cheval)
      * tete "place"     -> probabilite sigmoide independante (multi-tache)
  - Optimiseur Adam ecrit a la main, retropropagation manuelle.

Pourquoi NumPy et pas PyTorch/TensorFlow : l'environnement d'execution ne
dispose que d'1 CPU sans GPU, et les paquets PyTorch/TensorFlow standards
embarquent plusieurs Go de dependances CUDA inutiles ici. Le reseau ci-dessous
est un vrai reseau de neurones entraine par retropropagation ; il tourne en
quelques secondes sur ce volume de donnees (~40 000 lignes d'entrainement).
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class AdamOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class HorseRaceNet:
    def __init__(self, cat_cardinalities, cat_dims, n_numeric, hidden=(64, 32), seed=42):
        """
        cat_cardinalities: dict nom -> nb de categories (incl. OOV=0)
        cat_dims: dict nom -> dimension d'embedding
        n_numeric: nb de features numeriques continues
        """
        rng = np.random.default_rng(seed)
        self.cat_names = list(cat_cardinalities.keys())
        self.cat_dims = cat_dims
        self.params = {}

        for name in self.cat_names:
            card = cat_cardinalities[name]
            dim = cat_dims[name]
            self.params[f'emb_{name}'] = rng.normal(0, 0.05, size=(card, dim))

        emb_total = sum(cat_dims.values())
        in_dim = emb_total + n_numeric
        h1, h2 = hidden

        def glorot(fan_in, fan_out):
            limit = np.sqrt(6 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.params['W1'] = glorot(in_dim, h1)
        self.params['b1'] = np.zeros(h1)
        self.params['W2'] = glorot(h1, h2)
        self.params['b2'] = np.zeros(h2)
        self.params['W_win'] = glorot(h2, 1)
        self.params['b_win'] = np.zeros(1)
        self.params['W_place'] = glorot(h2, 1)
        self.params['b_place'] = np.zeros(1)

        self.n_numeric = n_numeric
        self.in_dim = in_dim

    def _embed_concat(self, cat_idx, num_x):
        """cat_idx: dict nom -> array(int) [N]; num_x: array [N, n_numeric]"""
        parts = []
        cache = {}
        for name in self.cat_names:
            idx = cat_idx[name]
            emb = self.params[f'emb_{name}'][idx]  # [N, dim]
            parts.append(emb)
            cache[name] = idx
        parts.append(num_x)
        x = np.concatenate(parts, axis=1)
        return x, cache

    def forward(self, cat_idx, num_x, dropout_p=0.0, training=False, rng=None):
        x, emb_cache = self._embed_concat(cat_idx, num_x)
        z1 = x @ self.params['W1'] + self.params['b1']
        a1 = relu(z1)
        mask1 = None
        if training and dropout_p > 0:
            mask1 = (rng.random(a1.shape) > dropout_p).astype(a1.dtype) / (1 - dropout_p)
            a1 = a1 * mask1

        z2 = a1 @ self.params['W2'] + self.params['b2']
        a2 = relu(z2)
        mask2 = None
        if training and dropout_p > 0:
            mask2 = (rng.random(a2.shape) > dropout_p).astype(a2.dtype) / (1 - dropout_p)
            a2 = a2 * mask2

        win_score = (a2 @ self.params['W_win'] + self.params['b_win']).ravel()      # [N] logit pre-softmax-course
        place_logit = (a2 @ self.params['W_place'] + self.params['b_place']).ravel()
        place_prob = sigmoid(place_logit)

        cache = dict(x=x, z1=z1, a1=a1, mask1=mask1, z2=z2, a2=a2, mask2=mask2,
                     emb_cache=emb_cache, cat_idx=cat_idx, num_x=num_x)
        return win_score, place_prob, cache

    def backward(self, cache, d_win_score, d_place_prob_logit, l2=1e-5):
        """d_win_score: gradient dLoss/d(win_score) [N] (deja la derivee softmax-crossentropy)
           d_place_prob_logit: gradient dLoss/d(place_logit) [N] (deja sigmoid-BCE derivee)"""
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        a2 = cache['a2']

        grads['W_win'] = a2.T @ d_win_score.reshape(-1, 1) + l2 * self.params['W_win']
        grads['b_win'] = d_win_score.sum(axis=0, keepdims=True).ravel()
        grads['W_place'] = a2.T @ d_place_prob_logit.reshape(-1, 1) + l2 * self.params['W_place']
        grads['b_place'] = d_place_prob_logit.sum(axis=0, keepdims=True).ravel()

        d_a2 = (d_win_score.reshape(-1, 1) @ self.params['W_win'].T +
                d_place_prob_logit.reshape(-1, 1) @ self.params['W_place'].T)
        if cache['mask2'] is not None:
            d_a2 = d_a2 * cache['mask2']
        d_z2 = d_a2 * relu_grad(cache['z2'])

        grads['W2'] = cache['a1'].T @ d_z2 + l2 * self.params['W2']
        grads['b2'] = d_z2.sum(axis=0)

        d_a1 = d_z2 @ self.params['W2'].T
        if cache['mask1'] is not None:
            d_a1 = d_a1 * cache['mask1']
        d_z1 = d_a1 * relu_grad(cache['z1'])

        grads['W1'] = cache['x'].T @ d_z1 + l2 * self.params['W1']
        grads['b1'] = d_z1.sum(axis=0)

        d_x = d_z1 @ self.params['W1'].T
        offset = 0
        for name in self.cat_names:
            dim = self.cat_dims[name]
            d_emb = d_x[:, offset:offset + dim]
            idx = cache['cat_idx'][name]
            np.add.at(grads[f'emb_{name}'], idx, d_emb)
            offset += dim

        return grads

    def save(self, path):
        np.savez(path, **self.params,
                 cat_names=np.array(self.cat_names, dtype=object),
                 cat_dims=np.array([self.cat_dims[n] for n in self.cat_names]),
                 n_numeric=self.n_numeric)

    @classmethod
    def load(cls, path, cat_cardinalities):
        data = np.load(path, allow_pickle=True)
        cat_names = list(data['cat_names'])
        cat_dims = {n: int(d) for n, d in zip(cat_names, data['cat_dims'])}
        n_numeric = int(data['n_numeric'])
        net = cls(cat_cardinalities, cat_dims, n_numeric)
        for k in net.params:
            net.params[k] = data[k]
        return net
