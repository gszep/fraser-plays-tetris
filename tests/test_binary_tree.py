from functools import partial

import jax
import jax.numpy as jnp
from genjax import Arguments, Const, categorical, flip, gen, normal, switch, uniform
from jax import jit, vmap
from jaxtyping import Array, Float, PRNGKeyArray

MAX_DEPTH = 2


@gen
def K1(node_id: Const[int]):
    y = normal(0.0, 1.0) @ f"normal:{node_id.unwrap()}"
    return y


@gen
def K2(node_id: Const[int]):
    y = uniform(0.0, 1.0) @ f"uniform:{node_id.unwrap()}"
    return y


@gen
def leaf(node_id: Const[int], depth: Const[int], *args: Float[Array, "..."]):

    branch_prob, kernel_prob = args
    kernel_logits = jnp.log(kernel_prob)

    kernel = categorical(kernel_logits) @ f"kernel:{node_id.unwrap()}"
    kernel_args = [(node_id,)] * len(kernel_logits)
    y = switch(K1, K2)(kernel, *kernel_args) @ f"leaf:{node_id.unwrap()}"

    return y


@gen
def branch(node_id: Const[int], depth: Const[int], *args: Float[Array, "..."]):

    x = model(Const(2 * node_id.unwrap()), Const(depth.unwrap() + 1), *args) @ f"branch:left:{node_id.unwrap()}"
    y = model(Const(2 * node_id.unwrap() + 1), Const(depth.unwrap() + 1), *args) @ f"branch:right:{node_id.unwrap()}"

    return x + y


@gen
def model(node_id: Const[int], depth: Const[int], *args: Float[Array, "..."]):
    branch_prob, kernel_prob = args
    model_args = (node_id, depth, *args)

    if depth.unwrap() >= MAX_DEPTH:
        value = leaf(*model_args) @ f"node_value:{node_id.unwrap()}"

    else:
        is_branch = flip(branch_prob) @ f"is_branch:{node_id.unwrap()}"
        value = branch.or_else(leaf)(is_branch, model_args, model_args) @ f"node_value:{node_id.unwrap()}"

    return value


@jit
@partial(vmap, in_axes=(0, None))
def generate_samples(key: PRNGKeyArray, args: Arguments):
    trace = model.simulate(key, args)
    samples = trace.get_sample()
    return samples


def test_binary_tree():
    keys = jax.random.split(jax.random.PRNGKey(42), 200)
    samples = generate_samples(
        keys,
        (
            Const(1),  # initial node_id
            Const(0),  # initial depth
            jnp.array(0.3),  # branch_prob
            jnp.array([0.5, 0.5]),  # kernel_prob
        ),
    )

    assert samples.shape == (200,)
