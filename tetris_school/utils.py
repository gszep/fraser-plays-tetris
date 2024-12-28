import random
from collections import deque, namedtuple

import matplotlib.pyplot as plt
from colorama import Fore, Style

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory: deque = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot(stats: dict, yscale: str = "linear") -> None:
    plt.figure()

    for label, x in stats.items():
        if isinstance(x, list):
            plt.plot(x, label=label)

    plt.xlabel("Episode")
    plt.yscale(yscale)
    plt.legend()

    plt.savefig("plot.png")
    plt.close()


GAME_OVER = -1
REWARD_UNICODE = {
    0: "–",
    -1: f"{Fore.YELLOW}▄{Style.RESET_ALL}",
    1: f"{Fore.CYAN}▀{Style.RESET_ALL}",
}
