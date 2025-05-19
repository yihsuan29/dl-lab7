#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
import os
import re

SUCCESS_REWARD_THRESHOLD = -150
NUM_SUCCESSFUL_SEEDS = 20

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
        hidden_dim: int = 64
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        # self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = init_layer_uniform(nn.Linear(hidden_dim, out_dim))
        self.log_std_layer = init_layer_uniform(nn.Linear(hidden_dim, out_dim))

        
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = self.layer1(state)
        x = F.relu(x)
        # x = self.layer2(x)
        # x = F.relu(x)
        mu = torch.tanh(self.mu_layer(x)) * 2 # action space [-2,2]
        log_std = torch.tanh(self.log_std_layer(x)) # softplus >=0
        log_std = self.log_std_min + (log_std + 1) * (self.log_std_max - self.log_std_min)/2
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        # self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = init_layer_uniform(nn.Linear(hidden_dim, 1))
        
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = self.layer1(state)
        x = F.relu(x)
        # x = self.layer2(x)
        # x = F.relu(x)
        value = self.out(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae.
       At = δt + γλδt + 1 + (γλ)^2δt+2 + · · ·
       At = δt + γλAt+1
       δt = rt + γV(s') - V(s)
    """
    # ############TODO#############
    gae = 0
    gae_returns = []
    next_v = next_value
    
    for reward, value, mask in  zip(reversed(rewards), reversed(values), reversed(masks)):
        delta = reward + gamma * next_v * mask - value
        gae = delta + gamma * tau * gae * mask
        gae_returns.insert(0,gae+value)
        next_v = value
        
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False
        
        self.best_score = float("-inf")
        self.model_dir = args.save_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, epoch: int = None, is_best: bool = False):
        if is_best:
            actor_path = os.path.join(self.model_dir, "actor_best.pt")
            critic_path = os.path.join(self.model_dir, "critic_best.pt")
            print(f"Saving best model with score {self.best_score:.2f} to {actor_path}")
        elif epoch is not None and epoch >= 150 and epoch % 10 == 0:
            actor_path = os.path.join(self.model_dir, f"actor_epoch_{epoch}.pt")
            critic_path = os.path.join(self.model_dir, f"critic_epoch_{epoch}.pt")
            print(f"Epoch {epoch}: Saving model to {actor_path}")
        else:
            return  

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()          

            # actor_loss
            ############TODO#############
            # actor_loss =  -min (ρ(θ)A(s, a), clip (ρ(θ), 1 − ϵ, 1 + ϵ) A(s, a))
            surrogate = ratio * adv
            clipped_surrogate = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)* adv
            
            entropy = dist.entropy().mean()
            actor_loss = (-torch.min(surrogate, clipped_surrogate)).mean() - self.entropy_weight*entropy          
            
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?            
            state_value = self.critic(state)
            critic_loss = F.mse_loss(state_value, return_)
            #critic_loss = F.smooth_l1_loss(state_value, return_) 
            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset()
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        for ep in tqdm(range(1, self.num_episodes)):
            score = 0
            print("\n")
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]        

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"Episode {episode_count}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "episode": episode_count,
                        "return": score
                        }, step=self.total_step)  
      
                    
                    if score >= self.best_score:
                        self.best_score = score
                        self.save_model(is_best=True)

                    self.save_model(epoch=episode_count)
                    score = 0             


            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            wandb.log({
                    "step": self.total_step,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    }, step=self.total_step) 

        # termination
        self.env.close()

    def test(self, video_folder: str, seed: int):
        """Test the agent."""
        self.is_test = True

        name_prefix = f"seed_{seed}"
        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, name_prefix=name_prefix)

        state, _ = self.env.reset(seed=seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score[0][0])
        self.env.close()
        self.env = tmp_env
        return score[0][0]
        
    
    def load_best_model(self, save_dir: str):
        actor_path = os.path.join(save_dir, "actor_best.pt")
        critic_path = os.path.join(save_dir, "critic_best.pt")        

        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        print(f"Loaded model from {actor_path}")
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def read_seeds_from_file(file_path: str):
    seeds = []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r"Seed:\s*(\d+),\s*Reward:", line)
            if match:
                seeds.append(int(match.group(1)))
    return seeds
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=5e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=float, default=20)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)  
    parser.add_argument("--update-epoch", type=float, default=64)
    parser.add_argument("--save-dir", type=str, default="result/task2/model", help="Directory to save model checkpoints")
    parser.add_argument("--video-dir", type=str, default="result/task2/video_test_seed", help="Directory to save test videos")
    parser.add_argument("--txt-dir", type=str, default="result/task2", help="Directory to save test seed")

    args = parser.parse_args()
 
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    # wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True)
    
    # agent = PPOAgent(env, args)
    # agent.train()
    
    # agent = PPOAgent(env, args)
    # os.makedirs(args.txt_dir, exist_ok=True)
    # result_file = os.path.join(args.txt_dir, "successful_seeds.txt")

    # os.makedirs(args.video_dir, exist_ok=True)
    # agent.load_best_model(save_dir=args.save_dir)

    # success_count = 0
    # with open(result_file, "w") as f:
    #     for i in range(10000):
    #         score = agent.test(video_folder=args.video_dir, seed=i)
    #         if score > SUCCESS_REWARD_THRESHOLD:
    #             f.write(f"Seed: {i}, Reward: {score:.2f}\n")
    #             print(f"Success {success_count + 1}/20: Seed {i}, Reward {score:.2f}")
    #             success_count += 1
    #             if success_count >= NUM_SUCCESSFUL_SEEDS:
    #                 break
    #         else:
    #             print(f"Failed: Seed {i}, Reward {score:.2f}")

    os.makedirs(args.video_dir, exist_ok=True)

    agent = PPOAgent(env, args)
    agent.load_best_model(save_dir=args.save_dir)
    file_path = "result/task2/successful_seeds.txt"
    seeds = read_seeds_from_file(file_path)
    for seed in seeds:
        print(f"Testing with seed {seed}")
        agent.test(video_folder=args.video_dir, seed=seed)