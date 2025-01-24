import math
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Optional, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pygame
import torch
from gymnasium import spaces
from jax import grad, jit, random, vmap
from jax.experimental.pallas import MemoryRef, load, pallas_call, program_id, store
from jaxtyping import Array, Bool, Float, Int32, PRNGKeyArray, Scalar, Shaped, UInt4
from torch.types import Device

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
RED2 = (255, 100, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GRAY2 = (100, 100, 100)

REWARD_COLOR = {1: (BLUE1, BLUE2), -1: (RED, RED2), 0: (GRAY, WHITE)}
BLOCK_SIZE = 20

BACKGROUND = 0
BLOCKS = 1
TETROMINO = 2


class Tetris(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1000}
    actions = Enum("actions", "NOTHING RIGHT LEFT ROTATE", start=0)
    action_space = spaces.Discrete(4)

    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        render_mode: Optional[str] = None,
        max_score: int = 100,
        device: Optional[Device] = None,
    ):
        self.device = device
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.max_score = max_score

        # define size of game board
        self.width = width
        self.height = height

        self.placedBlocks = torch.zeros((self.width, self.height), dtype=torch.int, device=self.device)
        self._reward = torch.tensor(0, dtype=torch.float, device=self.device)
        self.height_range = torch.arange(self.height, dtype=torch.int, device=self.device) + 1

        # define starting shape
        self._x = {
            ".": torch.tensor(
                [
                    self.width // 2,
                ],
                dtype=torch.int,
                device=self.device,
            ),
            "-": torch.tensor(
                [
                    self.width // 2,
                    self.width // 2,
                ],
                dtype=torch.int,
                device=self.device,
            ),
            "L": torch.tensor(
                [
                    self.width // 2,
                    self.width // 2 + 1,
                    self.width // 2,
                ],
                dtype=torch.int,
                device=self.device,
            ),
        }
        self._y = {
            ".": torch.tensor(
                [
                    self.height,
                ],
                dtype=torch.int,
                device=self.device,
            ),
            "-": torch.tensor(
                [
                    self.height,
                    self.height + 1,
                ],
                dtype=torch.int,
                device=self.device,
            ),
            "L": torch.tensor(
                [
                    self.height,
                    self.height,
                    self.height + 1,
                ],
                dtype=torch.int,
                device=self.device,
            ),
        }

        self.keys = ["-"]
        key = self.np_random.choice(self.keys)
        self.shape = {"x": self._x[key].clone(), "y": self._y[key].clone()}

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # needed to seed self.np_random

        self._new_shape()
        self.placedBlocks.mul_(0)

        self.iter = 0
        self.score = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        state = self.placedBlocks.clone()
        x, y = self.shape_inview

        state[x, y] = 2
        if self._inview.any():
            ymin = y.min(dim=-1)
            state[x[ymin.indices], ymin.values] = 3

        return state

    def _get_info(self):
        return {
            "shape": self.shape,
            "score": self.score,
        }

    def _clear_rows(self):
        isfull = self.placedBlocks.all(axis=0)  # type: ignore
        num_full = sum(isfull)

        updatedBlocks = self.placedBlocks[:, ~isfull]
        self.placedBlocks.fill_(value=0)

        self.placedBlocks[:, : self.height - num_full] = updatedBlocks
        self.score += num_full

    def _new_shape(self):
        key = self.np_random.choice(self.keys)
        self.shape["x"], self.shape["y"] = self._x[key].clone(), self._y[key].clone()

        for _ in range(np.random.randint(0, 4)):
            self.rotate_shape()

        return self.shape

    @property
    def terminated(self) -> torch.Tensor:
        return self.placedBlocks[:, -1].any()

    @property
    def truncated(self) -> bool:
        return self.score >= self.max_score

    @property
    def board_height(self) -> torch.Tensor:
        return (self.height_range * self.placedBlocks.any(dim=0)).argmax()

    def project_shape(self, x, y):
        y = y.clone()

        while y.min() > 0 and not self.placedBlocks[x, y - 1].any():
            y -= 1

        return x, y

    def step(self, action: Union[int, torch.Tensor]):
        self.reward = 0  # type: ignore

        # move shape
        x, y = self.move_shape(action)

        # reward based on projected shape
        if self._inview.all():
            xp, yp = self.project_shape(x, y)

            blocks = self.placedBlocks.clone()
            blocks[xp, yp] = 1

            if blocks.all(axis=0).any():  # type: ignore # reward for full rows
                self.reward += 1
            else:  # penalize increase in board height
                delta_height = yp.max() - self.board_height
                self.reward += 1 if delta_height <= 0 else -1

            # penalize for gaps under the shape
            if (blocks[xp, yp.min() - 1] == 0).any() and yp.min() > 0:
                self.reward -= 1

        # gravity
        if self.y.min() > 0:  # boundary check
            if not self.placedBlocks[x, y - 1].any():  # collision check
                self.y -= 1

        # place shape if it hits the bottom or other blocks
        x, y = self.shape_inview
        if y.min() == 0 or self.placedBlocks[x, y - 1].any():
            self.placedBlocks[x, y] = 1
            self._clear_rows()
            self._new_shape()

        done = self.done
        reward = self.reward.clone()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.iter += 1
        return observation, reward, done, info

    def move_shape(self, action: Union[int, torch.Tensor]):
        x, y = self.shape_inview

        # possible actions
        if action == self.actions.RIGHT.value:
            if self.x.max() < self.width - 1:  # boundary check
                if not self.placedBlocks[x + 1, y].any():  # collision check
                    self.x += 1

        elif action == self.actions.LEFT.value:
            if self.x.min() > 0:  # boundary check
                if not self.placedBlocks[x - 1, y].any():  # collision check
                    self.x -= 1

        elif action == self.actions.ROTATE.value:
            self.rotate_shape()

        return self.shape_inview

    def rotate_shape(self):

        x_median = self.x.median()
        y_median = self.y.median()

        # rotate shape
        x_rotated = x_median - (self.y - y_median)
        y_rotated = y_median + (self.x - x_median)

        if x_rotated.max() < self.width - 1 and x_rotated.min() > 0:  # boundary check
            # check for collisions
            if not self.placedBlocks[x_rotated[y_rotated < self.height], y_rotated[y_rotated < self.height]].any():

                self.x = x_rotated
                self.y = y_rotated

    @property
    def size(self):
        return self.width, self.height

    @property
    def done(self) -> Union[bool, torch.Tensor]:
        return self.terminated or self.truncated

    @property
    def reward(self) -> torch.Tensor:
        return self._reward

    @reward.setter
    def reward(self, value: float):
        self._reward.fill_(value)

    @property
    def _inview(self) -> torch.Tensor:
        return self.shape["y"] < self.height

    @property
    def shape_inview(self) -> tuple:
        return self.x[self._inview], self.y[self._inview]

    @property
    def x(self) -> torch.Tensor:
        """x index of shape points"""
        return self.shape["x"]

    @x.setter
    def x(self, value: torch.Tensor):
        self.shape["x"] = value

    @property
    def y(self) -> torch.Tensor:
        """y index of shape points"""
        return self.shape["y"]

    @y.setter
    def y(self, value: torch.Tensor):
        self.shape["y"] = value

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()

            self.font = pygame.font.SysFont("arial", 25)
            self.window = pygame.display.set_mode((BLOCK_SIZE * self.width, BLOCK_SIZE * self.height))  # type: ignore
            pygame.display.set_caption("Tetris")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()  # type: ignore

        canvas = pygame.Surface((BLOCK_SIZE * self.width, BLOCK_SIZE * self.height))
        canvas.fill(BLACK)

        for idx in self.placedBlocks.argwhere():
            x, y = idx

            pygame.draw.rect(canvas, GRAY2, pygame.Rect(x * BLOCK_SIZE, (self.height - y - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(canvas, GRAY, pygame.Rect(x * BLOCK_SIZE + 4, (self.height - y - 1) * BLOCK_SIZE + 4, 12, 12))

        PRIMARY, SECONDARY = REWARD_COLOR[np.sign(self.reward.item())]
        for x, y in zip(self.x, self.y):
            pygame.draw.rect(canvas, PRIMARY, pygame.Rect(x * BLOCK_SIZE, (self.height - y - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(canvas, SECONDARY, pygame.Rect(x * BLOCK_SIZE + 4, (self.height - y - 1) * BLOCK_SIZE + 4, 12, 12))

        text = self.font.render(f"Score: {self.score}", True, WHITE)

        # The following line copies our drawings from `canvas` to the visible window
        if self.window is not None:
            self.window.blit(text, [0, 0])
            self.window.blit(canvas, canvas.get_rect())

        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


@jax.tree_util.register_dataclass
@dataclass
class Environment:
    placed_blocks: UInt4[Array, "..."]
    x: Int32[Array, "4"]
    y: Int32[Array, "4"]


@jax.tree_util.register_dataclass
@dataclass
class State:
    key: PRNGKeyArray
    environment: Environment
    action: int | UInt4[Scalar, ""] = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    reward: float | Float[Scalar, ""] = field(default_factory=lambda: jnp.array(0, dtype=jnp.float32))
    done: bool | Bool[Scalar, ""] = field(default_factory=lambda: jnp.array(False, dtype=jnp.bool))


@jax.tree_util.register_dataclass
@dataclass
class StateBatch:
    key: Shaped[PRNGKeyArray, "2"]
    obs: UInt4[Array, "size ..."]
    action: UInt4[Array, "size"]
    reward: Float[Array, "size"]
    done: Bool[Array, "size"]


class JAXTetris(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 1000}
    actions = Enum("actions", "NOTHING RIGHT LEFT ROTATE", start=0)
    action_space = spaces.Discrete(4)

    screen: Optional[pygame.Surface] = None
    clock: Optional[pygame.time.Clock] = None
    colors = {BACKGROUND: BLACK, BLOCKS: WHITE, TETROMINO: BLUE1}

    def __init__(
        self,
        width: int = 7,
        height: int = 14,
        render_mode: Optional[str] = None,
        max_score: int = 100,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.max_score = max_score

        # define size of game board
        self.width = width
        self.height = height

        self.tetromino_size = 1

    @partial(jit, static_argnames=("self",))
    def step(self, state: State) -> State:
        state.reward = 0.0

        state.key, key = random.split(state.key)
        state.done = jax.random.uniform(key) > 0.5

        state.reward += state.action.astype(float)

        # apply gravity to tetromino
        state.environment.y -= 1

        if self.render_mode == "human":
            self._render_frame(state)

        return self.reset_conditional(state)  # type: ignore

    @partial(jit, static_argnames=("self",))
    def _get_obs(self, state: State) -> Array:
        y = pallas_call(
            self._obs_kernel,
            out_shape=jax.ShapeDtypeStruct((self.width, self.height), state.environment.placed_blocks.dtype),
            grid=(self.width, self.height),
        )(state.environment.placed_blocks, state.environment.x, state.environment.y)
        return y  # type: ignore

    @partial(jit, static_argnames=("self",))
    def _obs_kernel(self, placed_blocks: MemoryRef, x: MemoryRef, y: MemoryRef, output: MemoryRef):
        i, j = program_id(axis=0), program_id(axis=1)

        x, y = load(x, 0), load(y, 0)
        is_tetromino = (x == i) & (y == j)

        def _tetrimino(x: Array) -> Array:
            return TETROMINO + 0 * x  # needed for shape inference?

        def _not_tetrimino(x: Array) -> Array:
            return x

        pixel = jax.lax.cond(is_tetromino, _tetrimino, _not_tetrimino, load(placed_blocks, (i, j)))
        store(output, (i, j), pixel)

    @partial(jit, static_argnames=("self",))
    def reset_conditional(self, state: State) -> State:

        def _continue(_: int) -> State:
            return state

        state = jax.lax.cond(state.done, self.reset, _continue, state.key)
        return state

    @partial(jit, static_argnames=("self",))
    def reset(self, seed: Array) -> State:
        key, _ = random.split(seed)

        return State(
            key=key,
            environment=Environment(
                placed_blocks=jnp.zeros(shape=(self.width, self.height), dtype=jnp.int32),
                x=jnp.array([self.width // 2], dtype=jnp.int32),
                y=jnp.array([self.height // 2], dtype=jnp.int32),
            ),
            done=True,
        )

    def reset_batch(self, seed: PRNGKeyArray, size: int) -> StateBatch:
        return StateBatch(
            key=random.split(seed, size),
            obs=jnp.zeros((size, self.width, self.height), dtype=jnp.int32),
            action=jnp.zeros(size, dtype=jnp.int32),
            reward=jnp.zeros(size, dtype=jnp.float32),
            done=jnp.zeros(size, dtype=jnp.bool),
        )

    def _render_frame(self, state: State):
        if self.screen is None and self.render_mode == "human":

            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Tetris")

            self.screen = pygame.display.set_mode((BLOCK_SIZE * self.width, BLOCK_SIZE * self.height))
            self.surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

        if self.screen is not None:
            frame = np.zeros((*self.surface.get_size(), 3), dtype=int)

            frame[state.environment.placed_blocks == BLOCKS] = self.colors[BLOCKS]
            frame[state.environment.x, state.environment.y] = self.colors[TETROMINO]

            pygame.surfarray.blit_array(self.surface, frame[:, ::-1])
            pygame.transform.scale(self.surface, self.screen.get_size(), self.screen)

        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


def get_batch(key: Array, env: JAXTetris, size: int) -> StateBatch:

    state_batch: StateBatch = env.reset_batch(seed=key, size=size)
    state: State = env.reset(seed=key)

    body_fun = partial(step, env=env)

    if env.render_mode == "human":
        for i in range(0, size):
            state, state_batch = body_fun(i, (state, state_batch))
    else:
        (state, state_batch) = jax.lax.fori_loop(
            lower=0,
            upper=size,
            body_fun=body_fun,
            init_val=(state, state_batch),
        )
    return state_batch


def step(i, states: tuple[State, StateBatch], env: JAXTetris) -> tuple[State, StateBatch]:

    state, state_batch = states
    state_batch.obs = state_batch.obs.at[i].set(env._get_obs(state))

    state.action = random.randint(state_batch.key[i], (1,), 0, 2).astype(jnp.int32)[0]
    state_batch.action = state_batch.action.at[i].set(state.action)

    if env.render_mode == "human":
        with jax.disable_jit():
            state: State = env.step(state)
    else:
        state: State = env.step(state)

    state_batch.reward = state_batch.reward.at[i].set(state.reward)
    state_batch.done = state_batch.done.at[i].set(state.done)

    return state, state_batch


env = JAXTetris()
get_batches = jit(vmap(partial(get_batch, env=env, size=100)))

NUM_ENV = 500
keys = random.split(jax.random.PRNGKey(seed=0), NUM_ENV)
batch = get_batches(keys)

env = JAXTetris(render_mode="human")
get_batch(keys[0], env, size=256)
True
