import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_AGENTS = 50
B = 4
K = 6
TIME_LIMIT = 1000
EPSILON = 0.1
ALPHA = 0.9
GAMMA = 0.9

def plot_three_data(G, diff, local, x_label, y_label, title, num_agents, b, k):
    # Plot three agent rewards (G, Diff, and local)
    plt.figure(figsize=(10, 6))
    plt.plot(G, label='Global Rewards')
    plt.plot(local, label='Local Rewards')
    plt.plot(diff, label='Difference Rewards')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{title} Reward Comparison for {num_agents} Agents, b={b}, k={k}')
    plt.show()

def plot_histogram(x_axis, y_axis, num_agents):
    # Plot a histogram of a sample attendance profile
    plt.bar(x_axis, y_axis)
    plt.xlabel("Nights")
    plt.ylabel("Attendance")
    plt.title(f'Histogram of Sample Attendance Profile for {num_agents} Agents')
    plt.show()


def my_argmax(arr):
    max_value = np.max(arr)
    max_indices = np.flatnonzero(arr == max_value)
    return np.random.choice(max_indices)

# Reward Functions
def local_reward_function(x_k, b):
    return x_k * np.exp(-x_k / b)

# print(f'local_reward test: {local_reward_function(13,4)}')

# Agent Class
class Agent:
    def __init__(self, k):
        self.k = k
        self.q_action_values = np.zeros(k)

    def choose_action(self, last_week=None):
        if last_week:
            return my_argmax(self.q_action_values)

        if np.random.rand() <= EPSILON:
            return np.random.randint(self.k)
        else:
            # np.argmax function always returns 0 when elements have same value
            # return np.argmax(self.q_action_values)
            return my_argmax(self.q_action_values)

    # Simple value update
    def update_q_value(self, night, reward, old_q_table):
        if old_q_table is None:
            self.q_action_values[night] += ALPHA * (reward - self.q_action_values[night])
        else:
            self.temporal_difference = reward + (GAMMA * np.max(self.q_action_values)) - old_q_table
            self.q_action_values[night] = old_q_table + (ALPHA * self.temporal_difference)

# Environment Simulation
def simulate(num_agents, b, k):
    agents = [Agent(k) for _ in range(num_agents)]
    attendances = []
    set_old_q_table = []
    set_weekly_old_q_table = []

    for week in range(TIME_LIMIT):
        # print(f'Agent Q tables: {agents[0].q_action_values}')

        # Initialize attendance
        attendance = np.zeros(k)

        # Each agent selects an action (night to attend)
        actions = [agent.choose_action() for agent in agents]
        if week == TIME_LIMIT-2:
            last_week = week
            actions = [agent.choose_action(last_week) for agent in agents]

        # Count attendance for each night
        for action in actions:
            attendance[action] += 1

        # print(f'attendance: {attendance}')

        # Record attendances for histogram
        attendances.append(attendance)

        # print(f'agent action: {actions[0]} ')

        # Calculate system and local rewards
        weekly_local_rewards = local_reward_function(attendance, b)
        # print(f'weekly_local_rewards: {weekly_local_rewards}')

        # Compute each agent's rewards
        for i, agent in enumerate(agents):
            chosen_night = actions[i]
            local_reward = weekly_local_rewards[chosen_night]

            # Update agents' Q-values based on rewards with weekly old Q_table
            if len(set_weekly_old_q_table) != 0:
                agent.update_q_value(chosen_night, local_reward, set_weekly_old_q_table[week - 1][i])
            else:
                agent.update_q_value(chosen_night, local_reward, None)

            set_old_q_table.append(agents[i].q_action_values[chosen_night])

        # set_weekly_old_q_table[week][agent_i]
        set_weekly_old_q_table.append(set_old_q_table)

    return agents, actions, attendance, set_weekly_old_q_table

agents, actions, attendance, set_weekly_old_q_table = simulate(NUM_AGENTS, B, K)
print(f'attendance: {attendance}')
print(f'actions: {actions}')
print(f'set_weekly_old_q_table: {set_weekly_old_q_table[-1][0]}')
for i in range(NUM_AGENTS):
    #print(f'agent {i} q_table: {agents[i].q_action_values}')
    nash_action = np.argmax(agents[i].q_action_values)
    #if nash_action != actions[i]:
        #(f'Not Nash: {nash_action} vs {actions[i]}')

# print("Nash Equilibrium")


