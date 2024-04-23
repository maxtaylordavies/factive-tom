import time
from itertools import product

import jax
import jax.random as jr
import jax.numpy as jnp
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from factive_tom.functions import (
    predict_choice_belief,
    predict_choice_knowledge,
    is_logical_val,
)

sns.set_theme(style="darkgrid")


def get_keys():
    seed = int(time.time())
    s_key, a_key, d_key = jr.split(jr.PRNGKey(seed), 3)
    return seed, s_key, a_key, d_key


def compute_acc_vals(
    S,
    d_func,
    default_func,
    a_probs=jnp.linspace(0.0, 1.0, 10),
    d_probs=jnp.linspace(0.0, 1.0, 10),
    n_seeds=10,
    n_trials=1000,
):
    data = []
    for _ in tqdm(range(n_seeds)):
        seed, s_key, a_key, d_key = get_keys()
        for a_prob, d_prob in product(a_probs, d_probs):
            s_vals = jr.choice(s_key, S, shape=(n_trials,))
            a_vals = jr.bernoulli(a_key, a_prob, shape=(n_trials,)).astype(int)
            d_vals = jr.bernoulli(d_key, d_prob, shape=(n_trials,)).astype(int)

            b_preds = jax.vmap(
                predict_choice_belief, in_axes=(None, 0, 0, 0, None, None)
            )(S, s_vals, a_vals, d_vals, d_func, default_func)
            k_preds = jax.vmap(predict_choice_knowledge, in_axes=(None, 0, 0, 0, None))(
                S, s_vals, a_vals, d_vals, default_func
            )

            data.append(
                {
                    "seed": seed,
                    "P(access)": float(a_prob),
                    "P(distractor)": float(d_prob),
                    "Accuracy": float(jnp.mean(b_preds == k_preds)),
                }
            )

    return pd.DataFrame(data)


S = jnp.array([10, 20])  # state space
d_func = lambda S, s: 30  # distractor function
default_func = lambda S: 20  # default function

df = compute_acc_vals(S, d_func, default_func)

fig, ax = plt.subplots()
sns.lineplot(df, x="P(distractor)", y="Accuracy", hue="P(access)", ax=ax)
ax.set(title="Relative accuracy of factive ToM")
plt.show()
