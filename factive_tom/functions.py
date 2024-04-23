"""
Note: avoiding if statements because they don't play well with
jax.jit and jax.vmap, which we're using to speed things up.
"""

from typing import Any, Callable, Union, Sequence, Collection

from jax.experimental import checkify
import jax.numpy as jnp

# type for a function that outputs a default state value given a space of possible state values
DefaultFn = Callable[[Collection[int]], int]

# type for a function that outputs a false state given space of possible state values and the true state value
DistractorFn = Callable[[Collection[int], int], int]


def is_logical_val(x: Any) -> bool:
    return jnp.logical_or(x == 0, x == 1)


def compute_belief(
    S: Collection[int],
    s: int,
    a: int,
    d_present: int,
    d_func: DistractorFn,
    default_func: DefaultFn,
) -> int:
    """
    Compute an agent's belief about s as a function of perceptual access and possible distractor element

    Args:
        S (Collection[int]): the space of possible state values
        s (int): the true state of the world
        a (0 or 1): the agent's perceptual access to the state
        d_present (0 or 1): the presence of a distractor element
        d_func (DistractorFn): a function that outputs a false state induced by a distractor element
        default_func (DefaultFn): a function that outputs a default state value given a space of possible state values

    Returns:
        int: the agent's belief about the state of the world
    """
    checkify.check(is_logical_val(a), "perceptual access must be 0 or 1")
    checkify.check(is_logical_val(d_present), "distractor presence must be 0 or 1")

    no_access_case: int = (1 - a) * default_func(S)
    distractor_case: int = a * d_present * d_func(S, s)
    no_distractor_case: int = a * (1 - d_present) * s

    return no_access_case + distractor_case + no_distractor_case


def compute_knowledge(a: int, d_present: int) -> int:
    """
    Compute an agent's knowledge state about s as a function of perceptual access and possible distractor element

    Args:
        a (0 or 1): the agent's perceptual access to the world
        d_present (0 or 1): the presence of a distractor element

    Returns:
        0 or 1: whether or not the agent *knows* the true state of the world
    """
    checkify.check(is_logical_val(a), "perceptual access must be 0 or 1")
    checkify.check(is_logical_val(d_present), "distractor presence must be 0 or 1")
    return a * (1 - d_present)


def predict_choice_belief(
    S: Collection[int],
    s: int,
    a: int,
    d_present: int,
    d_func: DistractorFn,
    default_func: DefaultFn,
) -> int:
    """
    Predict an agent's choice by modelling their belief about s

    Args:
        S (Collection[int]): the space of possible state values
        s (int): the true state of the world
        a (0 or 1): the agent's perceptual access to the world
        d_present (0 or 1): the presence of a distractor element
        d_func (DistractorFn): a function that outputs a false state induced by a distractor element
        default_func (DefaultFn): a function that outputs a default state value given a space of possible state values

    Returns:
        int: the agent's prediction about the value of s
    """
    return compute_belief(S, s, a, d_present, d_func, default_func)


def predict_choice_knowledge(
    S: Collection[int], s: int, a: int, d_present: int, default_func: DefaultFn
) -> int:
    """
    Predict an agent's choice by modelling their knowledge state about s

    Args:
        S (Collection[int]): the space of possible state values
        s (int): the true state of the world
        a (0 or 1): the agent's perceptual access to the world
        d_present (0 or 1): the presence of a distractor element
        default_func (DefaultFn): a function that outputs a default state value given a space of possible state values
    """
    knows = compute_knowledge(a, d_present)
    checkify.check(is_logical_val(knows), "knowledge must be 0 or 1")
    return (knows * s) + ((1 - knows) * default_func(S))
