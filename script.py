import yfinance as yf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, lr, gamma, e_start, e_end, e_decay, state_space):
        self.state = None
        self.action = None
        self.lr = lr
        self.gamma = gamma
        self.epsilon = e_start
        self.epsilon_min = e_end
        self.epsilon_decay = e_decay
        self.q_table = {}
        self.state_space = state_space

    def get_action(self, state):
        state = self.discretize(state)
        if random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(['Buy', 'Sell', 'Hold'])
        else:
            q_values = [self.get_q_value(state, a) for a in ['Buy', 'Sell', 'Hold']]
            action = ['Buy', 'Sell', 'Hold'][np.argmax(q_values)]
        return action

    def discretize(self, state):
        return round(state*self.state_space)/self.state_space

    def get_q_value(self, state, action):
        state = self.discretize(state)
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    def update_q_value(self, reward, next_state):
        next_state = self.discretize(next_state)
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


class SarsaAgent(Agent):
    def update_q_value(self, reward, next_state, next_action):
        next_state = self.discretize(next_state)
        next_q_value = self.get_q_value(next_state, next_action)
        if (self.state, self.action) not in self.q_table:
            self.q_table[(self.state, self.action)] = 0
        self.q_table[(self.state, self.action)] += self.lr * (reward + self.gamma * next_q_value - self.get_q_value(self.state, self.action))


class ValueIterationAgent(Agent):
    def __init__(self, lr, gamma, state_space):
        super().__init__(lr, gamma, None, None, None, state_space)
        self.v_table = {}

    def init_v(self, state):
        state = self.discretize(state)
        if state not in self.v_table:
            self.v_table[state] = 0

    def get_action(self, state):
        state = self.discretize(state)
        q_values = [self.get_q_value(state, a) for a in ['Buy', 'Sell', 'Hold']]
        action = ['Buy', 'Sell', 'Hold'][np.argmax(q_values)]
        return action

    def update_q_value(self, state, action, reward, next_state):
        next_state = self.discretize(next_state)
        # We don't need to calculate q_value_next here, because it's calculated in the update_v_value function.
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        self.q_table[(state, action)] = reward + self.gamma * self.v_table[next_state]
    
    def update_v_value(self, state):
        state = self.discretize(state)
        v_value_next = max([self.get_q_value(state, a) for a in ['Buy', 'Sell', 'Hold']])
        self.v_table[state] = v_value_next


def simulate_trading(agent, data, money=10000, stock=0, agent_type='Q'):
    state = data[0]
    if agent_type == 'Value':
        agent.init_v(state)
    else:
        state = agent.discretize(state)
        agent.update_state_action(state, 'Hold')

    portfolio_values = []
    for i in range(len(data)):
        state = agent.discretize(state)
        action = agent.get_action(state)

        if action == 'Buy' and money >= state:
            stock += 1
            money -= state
        elif action == 'Sell' and stock > 0:
            stock -= 1
            money += state

        portfolio_value = money + stock
        portfolio_values.append(portfolio_value)
        
        if i == len(data)-1:
            reward = portfolio_value
        else:
            next_state = data[i+1]
            reward = money + stock*next_state - portfolio_value
        
        if agent_type == 'Q':
            agent.update_q_value(reward, next_state)
            agent.update_state_action(state, action)
            agent.update_epsilon()
        elif agent_type == 'Sarsa':
            next_action = agent.get_action(next_state)
            agent.update_q_value(reward, next_state, next_action)
            agent.update_state_action(state, action)
            agent.update_epsilon()
        elif agent_type == 'Value':
            agent.update_q_value(state, action, reward, next_state)
            agent.update_v_value(next_state)
            agent.update_state_action(state, action)

    return portfolio_values

# Create and train the agents
q_agent = Agent(lr=0.01, gamma=0.95, e_start=1.0, e_end=0.01, e_decay=0.995, state_space=100)
sarsa_agent = SarsaAgent(lr=0.01, gamma=0.95, e_start=1.0, e_end=0.01, e_decay=0.995, state_space=100)
value_iteration_agent = ValueIterationAgent(lr=0.01, gamma=0.95, state_space=100)

# Download SPY stock data
raw_data = yf.download('SPY','2000-01-01','2023-12-31')['Close']

# Normalize the data
data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())

# Split the data into training and testing data
train_data = data['2000-01-01':'2014-12-31']
test_data = data['2015-01-01':'2023-12-31']

# Train and test the Q-Learning agent
q_train_portfolio_values = simulate_trading(q_agent, train_data, agent_type='Q')
q_test_portfolio_values = simulate_trading(q_agent, test_data, agent_type='Q')

# Train and test the Sarsa agent
sarsa_train_portfolio_values = simulate_trading(sarsa_agent, train_data, agent_type='Sarsa')
sarsa_test_portfolio_values = simulate_trading(sarsa_agent, test_data, agent_type='Sarsa')

# Train and test the Value Iteration agent
value_iteration_train_portfolio_values = simulate_trading(value_iteration_agent, train_data, agent_type='Value')
value_iteration_test_portfolio_values = simulate_trading(value_iteration_agent, test_data, agent_type='Value')

# Plot the results
plt.figure(figsize=(12, 9))
plt.plot(q_test_portfolio_values, color='blue', label='Q-Learning Agent')
plt.plot(sarsa_test_portfolio_values, color='red', label='Sarsa Agent')
plt.plot(value_iteration_test_portfolio_values, color='green', label='Value Iteration Agent')
plt.title('Performance of the Q-Learning, Sarsa and Value Iteration Agents')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()