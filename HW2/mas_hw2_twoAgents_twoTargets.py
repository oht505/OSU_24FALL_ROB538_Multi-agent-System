import numpy as np
import matplotlib.pyplot as plt

# Gridworld parameters
ENV_ROWS = 5
ENV_COLS = 10
TARGET1_POS = (1,1)
TARGET2_POS = (4,8)
ACTIONS = ['up', 'right', 'down', 'left']

# Rewards setting
REWARDS = np.full((ENV_ROWS, ENV_COLS), -1)
REWARDS[1, 1] = 20
REWARDS[4, 8] = 20
# print(REWARDS)

# Initialize Q-table: shape (grid height, grid width, 4 actions)
Q_table_agent1 = np.zeros((ENV_ROWS, ENV_COLS, len(ACTIONS)))
Q_table_agent2 = np.zeros((ENV_ROWS, ENV_COLS, len(ACTIONS)))
# print(Q_table_agent1)
# print(Q_table_agent1)

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

def get_next_action(Q_table, state, epsilon):
    curr_row_idx, curr_col_idx = state

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

# Q-learning parameters
ALPHA = 0.9  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.9  # Initial exploration rate
NUM_EPISODES = 50000
set_reward_agent1 = []
set_reward_agent2 = []
sum_reward_agent1 = 0
sum_reward_agent2 = 0

# Q-Learning Algorithm
for episode in range(NUM_EPISODES):
    agent1_row_idx, agent1_col_idx = get_starting_location()
    agent2_row_idx, agent2_col_idx = get_starting_location()
    done_agent1 = done_agent2 = False
    move_agent1 = []
    move_agent2 = []

    while not (done_agent1 or done_agent2):

        move_agent1.append((agent1_row_idx, agent1_col_idx))
        move_agent2.append((agent2_row_idx, agent2_col_idx))

        agent1_action_idx = get_next_action(Q_table_agent1, (agent1_row_idx, agent1_col_idx), EPSILON)
        agent2_action_idx = get_next_action(Q_table_agent2, (agent2_row_idx, agent2_col_idx), EPSILON)

        agent1_old_row_idx, agent1_old_col_idx = agent1_row_idx, agent1_col_idx
        agent2_old_row_idx, agent2_old_col_idx = agent2_row_idx, agent2_col_idx

        agent1_row_idx, agent1_col_idx = get_next_location(agent1_row_idx, agent1_col_idx, agent1_action_idx)
        agent2_row_idx, agent2_col_idx = get_next_location(agent2_row_idx, agent2_col_idx, agent2_action_idx)

        reward_agent1 = REWARDS[agent1_row_idx, agent1_col_idx]
        if (agent1_row_idx, agent1_col_idx) == TARGET1_POS:
            reward_agent1 = 20
            done_agent1 = True
        else:
            reward_agent1 = -1

        reward_agent2 = REWARDS[agent2_row_idx, agent2_col_idx]
        if (agent2_row_idx, agent2_col_idx) == TARGET2_POS:
            reward_agent2 = 20
            done_agent2 = True
        else:
            reward_agent2 = -1

        sum_reward_agent1 += reward_agent1
        sum_reward_agent2 += reward_agent2

        old_Q_table_agent1 = Q_table_agent1[agent1_old_row_idx, agent1_old_col_idx, agent1_action_idx]
        temporal_difference_agent1 = reward_agent1 + (GAMMA * np.max(Q_table_agent1[agent1_row_idx, agent1_col_idx])) - old_Q_table_agent1

        old_Q_table_agent2 = Q_table_agent2[agent2_old_row_idx, agent2_old_col_idx, agent2_action_idx]
        temporal_difference_agent2 = reward_agent2 + (GAMMA * np.max(Q_table_agent2[agent2_row_idx, agent2_col_idx])) - old_Q_table_agent2

        new_Q_table_agent1 = old_Q_table_agent1 + (ALPHA * temporal_difference_agent1)
        Q_table_agent1[agent1_old_row_idx, agent1_old_col_idx, agent1_action_idx] = new_Q_table_agent1

        new_Q_table_agent2 = old_Q_table_agent2 + (ALPHA * temporal_difference_agent2)
        Q_table_agent2[agent2_old_row_idx, agent2_old_col_idx, agent2_action_idx] = new_Q_table_agent2

    set_reward_agent1.append(sum_reward_agent1)
    set_reward_agent2.append(sum_reward_agent2)

# print("Q_table Agent1: ")
# print(Q_table_agent1)
# print()
# print("Q_table Agent2: ")
# print(Q_table_agent2)
#
# print("move_agent1: ")
# print(move_agent1)
# print()
# print("move_agent2: ")
# print(move_agent2)

# print(set_reward_agent1)
# print(set_reward_agent2)
reward_both = 0
set_reward_both = []

for i in range(NUM_EPISODES):
    reward_both = set_reward_agent1[i] + set_reward_agent2[i]
    set_reward_both.append(reward_both)

#print(set_reward_both)
#
time = []
for t in range(NUM_EPISODES):
    time.append(t)

plt.figure(figsize=(10,6))
# plt.plot(time, set_reward_agent1, label="agent1", color='blue')
# plt.plot(time, set_reward_agent2, label="agent2", color='red')
plt.plot(time, set_reward_both, label="sum of the rewards of both agents", color='blue')
plt.title("Accumulative Reward")
plt.xlabel("Number of episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()