import numpy as np
import matplotlib.pyplot as plt

# Gridworld parameters
ENV_ROWS = 5
ENV_COLS = 10
# TARGET_POS = (1, 1)  # Position of the target T1
# START_POS = (2, 3)  # Starting position of the agent
ACTIONS = ['up', 'right', 'down', 'left']

# Rewards setting
REWARDS = np.full((ENV_ROWS, ENV_COLS), -1)
REWARDS[1, 1] = 20
# print(REWARDS)

# Initialize Q-table: shape (grid height, grid width, 4 actions)
Q_table = np.zeros((ENV_ROWS, ENV_COLS, 4))
# print(Q_table)

# Helper function to get action index
def is_terminal_state(curr_row_idx, curr_col_idx):
    if REWARDS[curr_row_idx, curr_col_idx] == -1.:
        return False
    else:
        return True

def get_starting_location():
    curr_row_idx, curr_col_idx = 2, 3
    while is_terminal_state(curr_row_idx, curr_col_idx):
        curr_row_idx, curr_col_idx = 2, 3
    return curr_row_idx, curr_col_idx

def get_next_action(curr_row_idx, curr_col_idx, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(Q_table[curr_row_idx, curr_col_idx])
    else:
        return np.random.randint(4)

def get_next_location(curr_row_idx, curr_col_idx, action_idx):
    new_row_idx = curr_row_idx
    new_col_idx = curr_col_idx

    if ACTIONS[action_idx] == 'up' and curr_row_idx > 0:
        new_row_idx -= 1
    elif ACTIONS[action_idx] == 'right' and curr_col_idx < ENV_COLS - 1:
        new_col_idx += 1
    elif ACTIONS[action_idx] == 'down' and curr_row_idx < ENV_ROWS - 1:
        new_row_idx += 1
    elif ACTIONS[action_idx] == 'left' and curr_col_idx > 0:
        new_col_idx -= 1

    return new_row_idx, new_col_idx

def get_shortest_path(start_row_idx, start_col_idx):
    if is_terminal_state(start_row_idx, start_col_idx):
        return []
    else:
        curr_row_idx, curr_col_idx = start_row_idx, start_col_idx
        shortest_path = []
        shortest_path.append([curr_row_idx, curr_col_idx])
        while not is_terminal_state(curr_row_idx, curr_col_idx):
            action_idx = get_next_action(curr_row_idx, curr_col_idx, 1.)
            curr_row_idx, curr_col_idx = get_next_location(curr_row_idx, curr_col_idx, action_idx)
            shortest_path.append([curr_row_idx, curr_col_idx])
        return shortest_path

# Q-learning parameters
ALPHA = 0.9  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.9  # Initial exploration rate
EPSILON_DECAY = 0.995
NUM_EPISODES = 100
set_reward = []
sum_reward = 0
set_oneEpi_move = []
set_moves = []

# Q-Learning Algorithm
for episode in range(NUM_EPISODES):
    row_idx, col_idx = get_starting_location()
    sum_reward = 0

    while not is_terminal_state(row_idx, col_idx):

        action_idx = get_next_action(row_idx, col_idx, EPSILON)

        old_row_idx, old_col_idx = row_idx, col_idx
        row_idx, col_idx = get_next_location(row_idx, col_idx, action_idx)

        reward = REWARDS[row_idx, col_idx]
        sum_reward += reward
        old_Q_table = Q_table[old_row_idx, old_col_idx, action_idx]
        # print(f'Q_table: {Q_table}')
        # print(f'old_Q_table: {old_Q_table}')
        # print(f'max_old_Q_table: {np.max(Q_table[row_idx, col_idx])}')
        temporal_difference = reward + (GAMMA * np.max(Q_table[row_idx, col_idx])) - old_Q_table

        new_Q_table = old_Q_table + (ALPHA * temporal_difference)
        Q_table[old_row_idx, old_col_idx, action_idx] = new_Q_table

        #set_oneEpi_move.append((row_idx,col_idx))
        #print(row_idx, col_idx)

    #set_moves.append(set_oneEpi_move)
    set_reward.append(sum_reward)

# print(set_moves)
# print(set_reward)
#
# time = []
# for t in range(NUM_EPISODES):
#     time.append(t)
#
# plt.figure(figsize=(10,6))
# plt.plot(time, set_reward, label="Reward", color='blue')
# plt.title("Rewards of Each Episode")
# plt.xlabel("Number of episode")
# plt.ylabel("Reward")
# plt.legend()
# plt.grid(True)
# plt.show()
