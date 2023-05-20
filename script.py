import yfinance as yf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, lr, gamma, e_start, e_end, e_decay):
        self.state = None
        self.action = None
        self.lr = lr
        self.gamma = gamma
        self.epsilon = e_start
        self.epsilon_min = e_end
        self.epsilon_decay = e_decay
        self.q_table = {}

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(['Buy', 'Sell', 'Hold'])
        else:
            q_values = [self.get_q_value(state, a) for a in ['Buy', 'Sell', 'Hold']]
            action = ['Buy', 'Sell', 'Hold'][np.argmax(q_values)]
        return action

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    def update_q_value(self, reward, next_state):
        q_values_next = [self.get_q_value(next_state, a) for a in ['Buy', 'Sell', 'Hold']]
        q_value_next = reward + self.gamma * max(q_values_next)
        if (self.state, self.action) not in self.q_table:
            self.q_table[(self.state, self.action)] = 0
        self.q_table[(self.state, self.action)] += self.lr * (q_value_next - self.get_q_value(self.state, self.action))

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_state_action(self, state, action):
        self.state = state
        self.action = action


def simulate_trading(agent, data, money=10000, stock=0):
    state = data[0]
    action = agent.get_action(state)
    agent.update_state_action(state, action)
    
    portfolio_values = []
    reward = 0

    for i in range(1, len(data)):
        next_state = data[i]
        action = agent.get_action(next_state)

        if action == 'Buy' and money >= next_state:
            stock += 1
            money -= next_state
        elif action == 'Sell' and stock > 0:
            stock -= 1
            money += next_state

        portfolio_values.append(money + stock*next_state)

        if i == len(data)-1:
            reward = money + stock*next_state

        agent.update_q_value(reward, next_state)
        agent.update_state_action(next_state, action)
        agent.update_epsilon()

    return portfolio_values

# Download SPY stock data
data = yf.download('SPY','2010-01-01','2023-12-31')['Close']

# Normalize the data
data = (data - data.min()) / (data.max() - data.min())

# Split the data into training and testing data
train_data = data['2010-01-01':'2020-12-31']
test_data = data['2021-01-01':'2023-12-31']

# Create and train the agent
agent = Agent(lr=0.01, gamma=0.95, e_start=1.0, e_end=0.01, e_decay=0.995)
train_portfolio_values = simulate_trading(agent, train_data)

# Test the agent
train_portfolio_values = [10000] + simulate_trading(agent, train_data)
test_portfolio_values = [10000] + simulate_trading(agent, test_data)

# Plot the training and testing portfolio values over time
plt.figure(figsize=(12,6))
plt.plot(train_data.index, train_portfolio_values, label='Training')
plt.plot(test_data.index, test_portfolio_values, label='Testing')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()