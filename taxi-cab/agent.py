import random
import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA: int = 6, eps: float = 0.4, alpha=0.2):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - eps: epsilon for epsilon greedy action
        - alpha: error influence
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps
        self.alpha = alpha


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space

        Select greedy action with probability epsilon, otherwise select an
        action randomly.
        """

        if random.random() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return random.choice(range(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if done: return
        self.Q[state][action] += self.alpha * (reward + np.max(self.Q[next_state]) -
                                               self.Q[state][action])

