import random
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class KArmBanditsConfig:
    k: int
    max_reward: float = 3.0
    min_reward: float = -3.0
    reward_std: float = 1.0
    epsilon: float = 0.0


@dataclass
class KArmNonstationaryBanditsConfig:
    k: int
    reward: float = 1.0
    reward_std: float = 0.5
    epsilon: float = 0.0


class Bandit:
    def __init__(self, reward, reward_std):
        self.true_reward = reward
        self.reward_std = reward_std

    def get_reward(self):
        return random.gauss(self.true_reward, self.reward_std)


class NonstationaryBandit(Bandit):
    def __init__(self, reward, reward_std):
        super().__init__(reward, reward_std)
        self.original_reward = reward

    def get_reward(self):
        return random.gauss(self.true_reward, self.reward_std)

    def random_walk(self):
        self.true_reward += random.gauss(0, 0.01)

    def reset_true_reward(self):
        self.true_reward = self.original_reward


def initialize_bandits(config: KArmBanditsConfig):
    bandits = []
    for i in range(config.k):
        bandit = Bandit(
            random.uniform(config.min_reward, config.max_reward), config.reward_std
        )
        bandits.append(bandit)
    return bandits


def initialize_nonstationary_bandits(config: KArmNonstationaryBanditsConfig):
    bandits = []
    for _ in range(config.k):
        bandits.append(NonstationaryBandit(config.reward, config.reward_std))
    return bandits


class KArmBandits:
    def __init__(self, k: int, epsilon: float, bandits):
        self.k = k
        self.epsilon = epsilon
        self.bandits = bandits

        self.q_estimates = [0.0 for _ in range(self.k)]
        self.action_counter = [0 for _ in range(self.k)]

    @property
    def optimal_action(self):
        return np.argmax([b.true_reward for b in self.bandits])

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.choice(range(self.k))

        # need to randomly choose among the tied best actions.
        max_q = max(self.q_estimates)
        best_actions = [i for i, q in enumerate(self.q_estimates) if q == max_q]
        return random.choice(best_actions)

    def estimate(self, action, reward):
        self.action_counter[action] += 1
        n = self.action_counter[action]
        prev = self.q_estimates[action]
        self.q_estimates[action] += (reward - prev) / n

    def play(self, steps: int = 1000):
        avg_reward = []
        optimal_action = self.optimal_action
        optimal_action_hits = []
        for _ in range(steps):
            action = self.choose_action()
            r = self.bandits[action].get_reward()
            self.estimate(action, r)
            avg_reward.append(r)
            optimal_action_hits.append(action == optimal_action)

        return avg_reward, optimal_action_hits

    def reset(self):
        self.q_estimates = [0.0 for _ in range(self.k)]
        self.action_counter = [0 for _ in range(self.k)]


class NonstationaryKArmBandits(KArmBandits):
    def __init__(self, k: int, epsilon: float, bandits, alpha=None):
        super().__init__(k, epsilon, bandits)
        # Determine how q_estimates is updated
        self.alpha = alpha

    def estimate(self, action, reward):
        self.action_counter[action] += 1
        n = self.action_counter[action]
        prev = self.q_estimates[action]
        if self.alpha is None:
            self.q_estimates[action] += (reward - prev) / n
        else:
            self.q_estimates[action] += (reward - prev) * self.alpha

    def play(self, steps: int = 1000):
        avg_reward = []
        optimal_action = self.optimal_action
        optimal_action_hits = []
        for _ in range(steps):
            action = self.choose_action()
            r = self.bandits[action].get_reward()
            self.estimate(action, r)
            avg_reward.append(r)
            optimal_action_hits.append(action == optimal_action)

            # all true rewards drifts
            for b in self.bandits:
                b.random_walk()

        return avg_reward, optimal_action_hits

    def reset(self):
        self.q_estimates = [0.0 for _ in range(self.k)]
        self.action_counter = [0 for _ in range(self.k)]
        # reset bandits' reward
        for b in self.bandits:
            b.reset_true_reward()


class KArmBanditsUCB(KArmBandits):
    def __init__(self, k: int, bandits, c=1):
        # epsilon is not used.
        super().__init__(k, epsilon=0.0, bandits=bandits)
        self.c = c

    def choose_action(self):
        untried = [i for i, cnt in enumerate(self.action_counter) if cnt == 0]
        if untried:
            return random.choice(untried)
        t = sum(self.action_counter)
        scores = [
            q + self.c * math.sqrt(math.log(t) / cnt)
            for q, cnt in zip(self.q_estimates, self.action_counter)
        ]
        max_score = max(scores)
        best_actions = [i for i, s in enumerate(scores) if s == max_score]
        return random.choice(best_actions)