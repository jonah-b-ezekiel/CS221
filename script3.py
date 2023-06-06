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

# Here the action space is simplified to only {0, 1} - not investing or investing all-in.
# State space is the past N days returns.

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
        state = str(state)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.actions, key=lambda x: self.q[state][x])

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        q_predict = self.q[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q[s_][self.choose_action(s_)]
        else:
            q_target = r
        self.q[s][a] += self.alpha * (q_target - q_predict)


class QLearning:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = defaultdict(lambda: defaultdict(lambda: 0.))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def choose_action(self, state):
        state = str(state)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.actions, key=lambda x: self.q[state][x])

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        q_predict = self.q[s][a]
        if s_ != 'terminal':
            if self.q[s_].values():  # Check if the sequence is not empty
                q_target = r + self.gamma * max(self.q[s_].values())
            else:
                q_target = r
        else:
            q_target = r
        self.q[s][a] += self.alpha * (q_target - q_predict)


class ValueIteration:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = defaultdict(lambda: defaultdict(lambda: 0.))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def choose_action(self, state):
        state = str(state)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.actions, key=lambda x: self.q[state][x])

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        q_predict = self.q[s][a]
        if s_ != 'terminal':
            if self.q[s_].values():  # Check if the sequence is not empty
                q_target = r + self.gamma * max(self.q[s_].values())
            else:
                q_target = r
        else:
            q_target = r
        self.q[s][a] += self.alpha * (q_target - q_predict)



# Now, initialize the ValueIteration agent along with QLearning and SARSA
value_iteration = ValueIteration(actions, epsilon, alpha, gamma)
q_learning = QLearning(actions, epsilon, alpha, gamma)
sarsa = SARSA(actions, epsilon, alpha, gamma)

agents = [q_learning, sarsa, value_iteration]

states = [training_data['daily_return'].values[i-N:i].tolist() for i in range(N, len(training_data['daily_return']))]
for i in range(N, len(training_data['daily_return'])):
    state = states[i-N]
    for agent in agents:
        action = agent.choose_action(state)
        reward = training_data['daily_return'].values[i] * action
        next_state = states[i-N+1] if i < len(training_data['daily_return']) - 1 else 'terminal'
        next_action = agent.choose_action(str(next_state)) if next_state != 'terminal' else 0
        agent.learn(state, action, reward, next_state)

# Create a time index for plotting
time_index = pd.concat([training_data, testing_data]).index[N:]

# Prepare your data arrays for portfolio values
portfolios_train = {'Q-Learning': [100.], 'SARSA': [100.], 'Value Iteration': [100.]}
portfolios_test = {'Q-Learning': [100.], 'SARSA': [100.], 'Value Iteration': [100.]}

# Populate the training data portfolios
for agent, portfolio in zip([q_learning, sarsa, value_iteration], portfolios_train.values()):
    for i in range(N, len(training_data['daily_return'])):
        state = states[i-N]
        action = agent.choose_action(state)
        reward = training_data['daily_return'].values[i] * action
        portfolio.append(portfolio[-1] * (1. + reward))

# Testing
test_states = [testing_data['daily_return'].values[i-N:i].tolist() for i in range(N, len(testing_data['daily_return']))]

# Populate the testing data portfolios
for agent, portfolio in zip([q_learning, sarsa, value_iteration], portfolios_test.values()):
    for i in range(N, len(testing_data['daily_return'])):
        state = test_states[i-N]
        action = agent.choose_action(state)
        reward = testing_data['daily_return'].values[i] * action
        portfolio.append(portfolio[-1] * (1. + reward))

# Start plotting
fig, ax = plt.subplots()

# Plot the training data portfolios
for agent, portfolio in portfolios_train.items():
    ax.plot(time_index[:len(portfolio)], portfolio, label=f'{agent} Train')

# Plot the testing data portfolios
for agent, portfolio in portfolios_test.items():
    ax.plot(time_index[-len(portfolio):], portfolio, label=f'{agent} Test')

# Decorate the plot
ax.set_xlabel('Year')
ax.set_ylabel('Portfolio Value')
ax.set_title("Trading Strategy Performance")
ax.legend()

plt.show()

