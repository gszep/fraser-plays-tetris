import math
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
from jaxtyping import Array
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


class JAXEnv(gym.Env):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0,))
    def step(self, state_key: tuple[Array, Array], action: Array) -> tuple[tuple[Array, Array], Array, Array, dict[str, Array]]:
        state, key = state_key

        state += 1
        done = action > 0.5
        reward = action

        state_key, info = self.reset_conditional((state, key), done)
        return state_key, reward, done, info

    def _get_info(self) -> dict[str, Array]:
        return {
            "score": jnp.zeros(shape=(1,)),
        }

    def _get_obs(self, state_key: tuple[Array, Array]) -> Array:
        state, _ = state_key
        return state

    def reset_conditional(self, state_key: tuple[Array, Array], done: Array) -> tuple[tuple[Array, Array], dict[str, Array]]:
        state, key = state_key

        def _continue(_) -> tuple[tuple[Array, Array], dict[str, Array]]:
            return (state_key, self._get_info())

        return jax.lax.cond(done, self.reset, _continue, key)  # type: ignore

    @partial(jit, static_argnums=(0,))
    def reset(self, key: Array) -> tuple[tuple[Array, Array], dict[str, Array]]:
        state = jnp.zeros(shape=(4,))
        key, _ = random.split(key)

        return (state, key), self._get_info()


@vmap
def run_episodes(key: Array) -> dict[str, Array]:

    batch = {
        "keys": random.split(key, MAX_STEPS),
        "state": jnp.zeros(shape=(MAX_STEPS, 4)),
        "reward": jnp.zeros(shape=(MAX_STEPS)),
        "done": jnp.zeros(shape=(MAX_STEPS), dtype=jnp.bool_),
    }

    state_key, _ = env.reset(key)
    (state_key, batch) = jax.lax.fori_loop(0, MAX_STEPS, step, (state_key, batch))
    return batch  # type: ignore


def step(i, store: tuple[tuple[Array, Array], dict[str, Array]]) -> tuple[tuple[Array, Array], dict[str, Array]]:
    state_key, batch = store
    action = random.randint(batch["keys"][i], (1,), 0, 2)[0]

    state_key, reward, done, info = env.step(state_key, action)
    state, key = state_key

    batch["state"] = batch["state"].at[i].set(state)
    batch["reward"] = batch["reward"].at[i].set(reward)
    batch["done"] = batch["done"].at[i].set(done)

    return state_key, batch


NUM_ENV = 7
MAX_STEPS = 100

keys = random.split(jax.random.PRNGKey(seed=0), NUM_ENV)
env = JAXEnv()

batch = run_episodes(keys)
done = batch["done"]
state = batch["state"][:, :, 0]
jnp.mean(batch["done"])
