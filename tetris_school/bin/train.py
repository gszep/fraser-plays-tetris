import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tetris_school.games import Tetris
from tetris_school.model import Fraser
from tetris_school.utils import GAME_OVER, REWARD_UNICODE, ReplayMemory, Transition, plot


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
    num_episodes: int = 200,
    batch_size: int = 128,
    ckpt_path: str = "model.ckpt",
    force: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = Tetris(render_mode="human" if ui else None, device=device)

    model = Fraser(
        num_actions=game.action_space.n,  # type: ignore
        embed_dim=64,
    ).to(device)

    if os.path.exists(ckpt_path) and not force:  # load pre-trained model
        model.load_state_dict(torch.load(ckpt_path))
        # temperature = min_temperature

    model_prime = copy.deepcopy(model).to(device)
    model_prime.load_state_dict(model.state_dict())

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    memory = ReplayMemory(memory_size)

    record = 0
    stats: dict[str, list] = {"score": [], "temperature": []}
    for i in range(num_episodes):
        print(f"Episode {i}\t[", end="")

        state, info = game.reset()
        while not game.done:

            # sample action from model
            with torch.no_grad():
                logits = model(state.unsqueeze(0))
                action = F.softmax(logits / temperature, dim=-1).argmax().squeeze()

            # anneal temperature
            temperature = max(temperature * anneal_factor, min_temperature)

            # perform action and get reward
            next_state, reward, terminated, truncated, info = game.step(action)

            if terminated:  # game over state
                next_state.fill_(GAME_OVER)

            # save transition to memory
            memory.push(state, action, next_state, reward)

            # visualize game rewards
            print(REWARD_UNICODE[np.sign(reward.item())], end="")
            if game.done:
                print("]")

            # move to next state
            state = next_state
            if len(memory) < batch_size:
                continue  # wait until we have enough transitions

            # sample batch from memory
            batch = Transition(*zip(*memory.sample(batch_size)))

            states = torch.stack(batch.state)
            actions = torch.stack(batch.action)
            next_states = torch.stack(batch.next_state)
            rewards = torch.stack(batch.reward)

            # compute Q(s_{t}, a)
            predicted_rewards = model(states).gather(1, actions.unsqueeze(1)).squeeze()

            # compute V(s_{t+1})
            with torch.no_grad():
                not_done = (next_states.view(batch_size, -1) != GAME_OVER).all(axis=1)  # type: ignore
                next_rewards = not_done * model_prime(next_states).max(dim=1).values

            rewards += gamma * next_rewards

            # huber loss for Q-learning
            loss = F.smooth_l1_loss(predicted_rewards, rewards)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()

            # in-place gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()

            # soft update of model_prime
            for param, param_prime in zip(model.parameters(), model_prime.parameters()):
                param_prime.data.copy_(tau * param.data + (1.0 - tau) * param_prime.data)

            if game.done:  # collect and plot stats
                stats["score"].append(game.score.item())  # type: ignore
                stats["temperature"].append(temperature)

                if game.score > record:
                    record = game.score.item()  # type: ignore
                    torch.save(model.state_dict(), ckpt_path)

                plot(stats, yscale="symlog")


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
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--ui", action="store_true", help="Render the game")
    parser.add_argument("--ckpt_path", type=str, default="model.ckpt", help="Checkpoint path")
    parser.add_argument("--force", action="store_true", help="Force to overwrite checkpoint")

    train(**vars(parser.parse_args()))
