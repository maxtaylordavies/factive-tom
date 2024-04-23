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
    s_key, a_key, d_key = jr.split(jr.key(seed), 3)
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
            k_preds_without_d = jax.vmap(
                predict_choice_knowledge, in_axes=(None, 0, 0, None, None)
            )(S, s_vals, a_vals, 0, default_func)
            k_preds_with_d = jax.vmap(
                predict_choice_knowledge, in_axes=(None, 0, 0, 0, None)
            )(S, s_vals, a_vals, d_vals, default_func)

            acc_without_d = float(jnp.mean(k_preds_without_d == b_preds))
            acc_with_d = float(jnp.mean(k_preds_with_d == b_preds))

            tmp = {
                "seed": seed,
                "P(access)": float(a_prob),
                "P(distractor)": float(d_prob),
            }

            data.append({**tmp, "d": False, "Accuracy": acc_without_d})
            data.append({**tmp, "d": True, "Accuracy": acc_with_d})

    return pd.DataFrame(data)


def make_plots(df, title, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    flare = sns.color_palette("flare", as_cmap=True)
    crest = sns.color_palette("crest", as_cmap=True)
    sns.lineplot(
        df, x="P(distractor)", y="Accuracy", hue="P(access)", ax=ax[0], palette=flare
    )
    sns.lineplot(
        df, x="P(access)", y="Accuracy", hue="P(distractor)", ax=ax[1], palette=crest
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)


S = jnp.array([10, -10])  # state space
d_func = lambda S, s: -s  # distractor function
default_func = lambda S: 10  # default function

df = compute_acc_vals(S, d_func, default_func)
make_plots(
    df[df["d"] == False],
    "Relative accuracy of factive ToM (d-unaware)",
    "accuracy_d_unaware.pdf",
)
make_plots(
    df[df["d"] == True],
    "Relative accuracy of factive ToM (d-aware)",
    "accuracy_d_aware.pdf",
)
