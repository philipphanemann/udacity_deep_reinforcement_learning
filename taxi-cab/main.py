from itertools import product
from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
epss = (0.0, 0.1, 0.2)
alphas = (0.05, 0.1, 0.2)
rewards = dict()
rewards_over_time = dict()

products = list(product(epss, alphas))
for i, (eps, alpha) in enumerate(products):
    print(f'{i}/{len(products)}:, eps: {eps}, alpha: {alpha}')
    agent = Agent(eps=eps, alpha=alpha)
    avg_rewards, best_avg_reward = interact(env, agent, num_episodes=20_000)
    rewards[(eps, alpha)] = best_avg_reward
    rewards_over_time[(eps, alpha)] = avg_rewards

