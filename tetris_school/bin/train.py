import argparse
import os
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, random, vmap
from jaxtyping import Array

from tetris_school.games import Env, State, Tetris


@jax.tree_util.register_dataclass
@dataclass
class Batch:
    reward: Array


def get_batch(key: Array, env: Env, model, size: int) -> Batch:
    batch = Batch(
        jnp.zeros(size, dtype=jnp.int32),
    )

    state = env.reset(seed=key)
    body_fun = partial(accumulate, env=env, model=model)

    if env.render_mode == "human":
        for i in range(0, size):
            state, batch = body_fun(i, (state, batch))
    else:
        (state, batch) = jax.lax.fori_loop(
            lower=0,
            upper=size,
            body_fun=body_fun,
            init_val=(state, batch),
        )
    return batch


def accumulate(i, states: tuple[State, Batch], env: Env, model) -> tuple[State, Batch]:
    state, batch = states

    state.action = model(state)
    state = env.step(state)

    if env.render_mode == "human":
        with jax.disable_jit():
            env._render(state)

    batch.reward = batch.reward.at[i].set(state.reward)
    return state, batch


def train(
    learning_rate: float = 1e-4,
    tau: float = 0.005,
    temperature: float = 1.0,
    anneal_factor: float = 0.5,
    min_temperature: float = 0.05,
    gamma: float = 0.99,
    ui: bool = False,
    num_workers: int = 1,
    memory_size: int = 10000,
    num_steps: int = 1024,
    batch_size: int = 128,
    ckpt_path: str = "model.ckpt",
    force: bool = False,
):
    game = Tetris(render_mode="human" if ui else None)

    @jit
    def model(state: State) -> Array:
        return random.randint(state.key, (1,), 0, 4).astype(jnp.int32)[0]

    if ui:  # retrieve only one batch for rendering
        non_jit_fn = partial(get_batch, env=game, model=model, size=num_steps)

        def get_batches(keys: Array, batch_idx: Optional[int] = 0) -> Batch:
            return non_jit_fn(keys[batch_idx])

    else:
        jit_fn = jit(vmap(partial(get_batch, env=game, model=model, size=num_steps)))

        def get_batches(keys: Array, batch_idx: Optional[int] = None) -> Batch:
            return jit_fn(keys)  # type: ignore

    keys = random.split(jax.random.PRNGKey(seed=0), num=batch_size)
    batch = get_batches(keys)

    print(batch.reward.sum(axis=-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tetris agent")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--tau", type=float, default=0.005, help="Tau for soft update")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for action sampling")
    parser.add_argument("--anneal_factor", type=float, default=0.9999, help="Annealing factor")
    parser.add_argument("--min_temperature", type=float, default=0.05, help="Minimum temperature")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers")
    parser.add_argument("--memory_size", type=int, default=10000, help="Memory size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=1024, help="Number of environment steps for a batch")
    parser.add_argument("--ui", action="store_true", help="Render the game")
    parser.add_argument("--ckpt_path", type=str, default="model.ckpt", help="Checkpoint path")
    parser.add_argument("--force", action="store_true", help="Force to overwrite checkpoint")

    train(**vars(parser.parse_args()))
