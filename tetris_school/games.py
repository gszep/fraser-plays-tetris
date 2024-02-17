from typing import Optional, Union
from enum import Enum
import torch
from torch.types import Device

import pygame
import gymnasium as gym
from gymnasium import spaces

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
RED2 = (255, 100, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

BLOCK_SIZE = 20


class Tetris(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}
    actions = Enum("Actions", "NOTHING RIGHT LEFT ROTATE", start=0)

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
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

        # we have 4 actions, corresponding to "nothing", "right", "left", "rotate"
        self.action_space = spaces.Discrete(4)

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

        self.keys = ["."]
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
        isfull = self.placedBlocks.all(axis=0)
        num_full = sum(isfull)

        updatedBlocks = self.placedBlocks[:, ~isfull]
        self.placedBlocks.fill_(value=0)

        self.placedBlocks[:, : self.height - num_full] = updatedBlocks
        self.score += num_full

    def _new_shape(self):
        key = self.np_random.choice(self.keys)
        self.shape["x"], self.shape["y"] = self._x[key].clone(), self._y[key].clone()
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
        board_height = self.board_height

        # move shape
        x, y = self.move_shape(action)

        # reward based on projected shape
        if self._inview.all():
            xp, yp = self.project_shape(x, y)

            # penalize increase in board height
            self.reward -= yp.max() - self.board_height

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

        # penalize increase in board height / reward for clearing rows
        self.reward -= self.board_height - board_height

        terminated = self.terminated
        reward = self.reward.clone()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.iter += 1
        truncated = self.truncated
        return observation, reward, terminated, truncated, info

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
            self.window = pygame.display.set_mode((BLOCK_SIZE * self.width, BLOCK_SIZE * self.height))
            pygame.display.set_caption("Tetris")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((BLOCK_SIZE * self.width, BLOCK_SIZE * self.height))
        canvas.fill(BLACK)

        for idx in self.placedBlocks.argwhere():
            x, y = idx

            pygame.draw.rect(canvas, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height - y - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(canvas, BLUE2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height - y - 1) * BLOCK_SIZE + 4, 12, 12))

        for x, y in zip(self.x, self.y):
            pygame.draw.rect(canvas, RED, pygame.Rect(x * BLOCK_SIZE, (self.height - y - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(canvas, RED2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height - y - 1) * BLOCK_SIZE + 4, 12, 12))

        x, y = self.x[0], self.y[0]
        pygame.draw.rect(canvas, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height - y - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render(f"Score: {self.score}", True, WHITE)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(text, [0, 0])
        self.window.blit(canvas, canvas.get_rect())

        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
