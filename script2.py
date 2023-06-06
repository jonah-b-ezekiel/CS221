import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Data Preprocessing
data = yf.download('AAPL', start='2000-01-01', end='2023-01-01')
data['daily_return'] = data['Close'].pct_change()

training_data = data.loc['2000-01-01':'2015-12-31']
testing_data = data.loc['2016-01-01':]

N = 5
actions = [0, 1]
epsilon = 0.1
alpha = 0.5
gamma = 0.95

class SARSA:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = defaultdict(lambda: defaultdict(lambda: 0.))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.actions, key=lambda x: self.q[state][x])

    def learn(self, s, a, r, s_, a_):
        q_predict = self.q[str(s)][a]
        q_target = r + self.gamma * self.q[str(s_)][a_]
        self.q[str(s)][a] += self.alpha * (q_target - q_predict)

class QLearning:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = defaultdict(lambda: defaultdict(lambda: 0.))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.actions, key=lambda x: self.q[state][x])

    def learn(self, s, a, r, s_):
        a_ = self.choose_action(str(s_))
        q_predict = self.q[str(s)][a]
        q_target = r + self.gamma * self.q[str(s_)][a_]
        self.q[str(s)][a] += self.alpha * (q_target - q_predict)

class ValueIteration:
    def __init__(self, actions, epsilon, alpha, gamma, theta=0.0001):
        self.v = defaultdict(lambda: 0.)
        self.q = defaultdict(lambda: defaultdict(lambda: 0.))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.actions = actions

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.actions, key=lambda x: self.q[state][x])

    def learn(self, s, a, r, s_):
        old_v = 0
        while True:
            delta = 0
            for s in states:
                if s == 'terminal':
                    continue
                action = self.choose_action(s)
                self.q[s][action] = r + self.gamma * self.v[s_]
                v = max(self.q[s][action] for action in self.actions)
                delta = max(delta, abs(v - self.v[s]))
                self.v[s] = v
            if delta < self.theta:
                break

# Training
q_learning = QLearning(actions, epsilon, alpha, gamma)
sarsa = SARSA(actions, epsilon, alpha, gamma)
value_iteration = ValueIteration(actions, epsilon, alpha, gamma)

agents = [q_learning, sarsa, value_iteration]

states = [str(training_data['daily_return'].values[i-N:i]) for i in range(N, len(training_data['daily_return']))]
for i in range(N, len(training_data['daily_return'])):
    state = states[i-N]
    for agent in agents:
        action = agent.choose_action(state)
        reward = training_data['daily_return'].values[i] * action
        next_state = states[i-N+1] if i < len(training_data['daily_return']) - 1 else 'terminal'
        if agent == q_learning:
            agent.learn(state, action, reward, next_state)
        elif agent == sarsa:
            next_action = agent.choose_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)
        elif agent == value_iteration:
            agent.learn(state, action, reward, next_state)

# Testing and plotting
test_states = [str(testing_data['daily_return'].values[i-N:i]) for i in range(N, len(testing_data['daily_return']))]
portfolios = {'Q-Learning': [100.], 'SARSA': [100.], 'Value Iteration': [100.]}

for agent, portfolio in zip([q_learning, sarsa, value_iteration], portfolios.values()):
    for i in range(N, len(testing_data['daily_return'])):
        state = test_states[i-N]
        action = agent.choose_action(state)
        portfolio.append(portfolio[-1] * (1. + action * testing_data['daily_return'].values[i]))

for agent, portfolio in portfolios.items():
    plt.plot(portfolio, label=agent)

plt.title("Trading Strategy Performance")
plt.legend()
plt.show()
