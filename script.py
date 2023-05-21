import yfinance as yf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class PolicyAgent:
    def __init__(self, target_allocation):
        self.target_allocation = target_allocation
        self.action = None

    def get_action(self, money, stock, price):
        if stock * price + money == 0:  # to prevent division by zero
            return 'Hold'
        current_allocation = (stock * price) / (money + stock * price)
        if current_allocation < self.target_allocation:
            return 'Buy'
        elif current_allocation > self.target_allocation:
            return 'Sell'
        else:
            return 'Hold'

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


def simulate_trading(agent, data, money=10000, stock=0, agent_type='Q'):
    state = data[0]
    if agent_type == 'Q':
        action = agent.get_action(state)
        agent.update_state_action(state, action)
    portfolio_values = []
    reward = 0
    for i in range(1, len(data)):
        next_state = data[i]
        if agent_type == 'Q':
            action = agent.get_action(next_state)
        else:
            action = agent.get_action(money, stock, next_state)
        if action == 'Buy' and money >= next_state:
            stock += 1
            money -= next_state
        elif action == 'Sell' and stock > 0:
            stock -= 1
            money += next_state
        portfolio_values.append(money + stock*next_state)
        if i == len(data)-1:
            reward = money + stock*next_state
        if agent_type == 'Q':
            agent.update_q_value(reward, next_state)
            agent.update_state_action(next_state, action)
            agent.update_epsilon()
    return portfolio_values

# Download SPY stock data
raw_data = yf.download('SPY','2000-01-01','2023-12-31')['Close']

# Normalize the data
data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())

# Split the data into training and testing data
train_data = data['2000-01-01':'2014-12-31']
test_data = data['2015-01-01':'2023-12-31']

# Create and train the agent
q_agent = Agent(lr=0.01, gamma=0.95, e_start=1.0, e_end=0.01, e_decay=0.995)
policy_agent = PolicyAgent(target_allocation=0.6)  # adjust the target allocation as needed

# Train and test the Q-Learning agent
q_train_portfolio_values = simulate_trading(q_agent, train_data, agent_type='Q')
q_test_portfolio_values = simulate_trading(q_agent, test_data, agent_type='Q')

# Train and test the policy-based agent
policy_train_portfolio_values = simulate_trading(policy_agent, train_data, agent_type='Policy')
policy_test_portfolio_values = simulate_trading(policy_agent, test_data, agent_type='Policy')

# Append the initial portfolio value to the beginning of portfolio values
q_train_portfolio_values = [10000] + q_train_portfolio_values
q_test_portfolio_values = [10000] + q_test_portfolio_values
policy_train_portfolio_values = [10000] + policy_train_portfolio_values
policy_test_portfolio_values = [10000] + policy_test_portfolio_values

# Create new date indices for the portfolio values
train_portfolio_index = pd.date_range(start=train_data.index[0], periods=len(q_train_portfolio_values))
test_portfolio_index = pd.date_range(start=test_data.index[0], periods=len(q_test_portfolio_values))

# Convert to DataFrame for easier manipulation
df_q_train_portfolio = pd.DataFrame(q_train_portfolio_values, index=train_portfolio_index, columns=['Q-Agent Portfolio'])
df_q_test_portfolio = pd.DataFrame(q_test_portfolio_values, index=test_portfolio_index, columns=['Q-Agent Portfolio'])
df_policy_train_portfolio = pd.DataFrame(policy_train_portfolio_values, index=train_portfolio_index, columns=['Policy-Agent Portfolio'])
df_policy_test_portfolio = pd.DataFrame(policy_test_portfolio_values, index=test_portfolio_index, columns=['Policy-Agent Portfolio'])

# Normalize portfolio values
df_q_train_portfolio['Q-Agent Portfolio'] = (df_q_train_portfolio['Q-Agent Portfolio'] - df_q_train_portfolio['Q-Agent Portfolio'].min()) / (df_q_train_portfolio['Q-Agent Portfolio'].max() - df_q_train_portfolio['Q-Agent Portfolio'].min())
df_q_test_portfolio['Q-Agent Portfolio'] = (df_q_test_portfolio['Q-Agent Portfolio'] - df_q_test_portfolio['Q-Agent Portfolio'].min()) / (df_q_test_portfolio['Q-Agent Portfolio'].max() - df_q_test_portfolio['Q-Agent Portfolio'].min())
df_policy_train_portfolio['Policy-Agent Portfolio'] = (df_policy_train_portfolio['Policy-Agent Portfolio'] - df_policy_train_portfolio['Policy-Agent Portfolio'].min()) / (df_policy_train_portfolio['Policy-Agent Portfolio'].max() - df_policy_train_portfolio['Policy-Agent Portfolio'].min())
df_policy_test_portfolio['Policy-Agent Portfolio'] = (df_policy_test_portfolio['Policy-Agent Portfolio'] - df_policy_test_portfolio['Policy-Agent Portfolio'].min()) / (df_policy_test_portfolio['Policy-Agent Portfolio'].max() - df_policy_test_portfolio['Policy-Agent Portfolio'].min())

# Plot the training and testing portfolio values over time
plt.figure(figsize=(12,6))
plt.plot(raw_data.index, data, label='S&P 500', color='grey')
plt.plot(df_q_train_portfolio.index, df_q_train_portfolio['Q-Agent Portfolio'], label='Q-Agent Training Portfolio')
plt.plot(df_q_test_portfolio.index, df_q_test_portfolio['Q-Agent Portfolio'], label='Q-Agent Testing Portfolio')
plt.plot(df_policy_train_portfolio.index, df_policy_train_portfolio['Policy-Agent Portfolio'], label='Policy-Agent Training Portfolio')
plt.plot(df_policy_test_portfolio.index, df_policy_test_portfolio['Policy-Agent Portfolio'], label='Policy-Agent Testing Portfolio')
plt.title('Normalized Portfolio Value and S&P 500 Over Time')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()