import numpy as np
import matplotlib.pyplot as plt

# Gridworld parameters
ENV_ROWS = 5
ENV_COLS = 10
TARGET1_POS = (1, 1)
TARGET2_POS = (4, 8)
ACTIONS = ['up', 'right', 'down', 'left']

# Rewards setting
REWARDS = np.full((ENV_ROWS, ENV_COLS), -1)
REWARDS[1, 1] = 20
REWARDS[4, 8] = 20
REWARD_CAPTURE_BOTH = 40
REWARD_CAPTURE_ONE = 20
# print(REWARDS)

# Initialize Q-table: shape (grid height, grid width, 4 actions)
Q_table_joint = np.zeros((ENV_ROWS, ENV_COLS, ENV_ROWS, ENV_COLS, len(ACTIONS), len(ACTIONS)))
# print(Q_table_joint)


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

def get_next_action(state, epsilon):
    x1, y1, x2, y2 = state

    if np.random.random() < epsilon:
        action_idx = np.unravel_index(np.argmax(Q_table_joint[x1,y1,x2,y2]), (len(ACTIONS), len(ACTIONS)))
        #print(action_idx)
        return ACTIONS[action_idx[0]], ACTIONS[action_idx[1]]
    else:
        return ACTIONS[np.random.randint(4)], ACTIONS[np.random.randint(4)]

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
set_global_reward = []
sum_global_reward = 0

# Q-Learning Algorithm
for episode in range(NUM_EPISODES):
    agent1_row_idx, agent1_col_idx = get_starting_location()
    agent2_row_idx, agent2_col_idx = get_starting_location()
    done_agent = False

    while not done_agent:
        state = (agent1_row_idx, agent1_col_idx, agent2_row_idx, agent2_col_idx)

        agent1_action_idx, agent2_action_idx = get_next_action(state, EPSILON)
        agent1_action_idx = ACTIONS.index(agent1_action_idx)
        agent2_action_idx = ACTIONS.index(agent2_action_idx)

        agent1_old_row_idx, agent1_old_col_idx = agent1_row_idx, agent1_col_idx
        agent2_old_row_idx, agent2_old_col_idx = agent2_row_idx, agent2_col_idx

        agent1_row_idx, agent1_col_idx = get_next_location(agent1_row_idx, agent1_col_idx, agent1_action_idx)
        agent2_row_idx, agent2_col_idx = get_next_location(agent2_row_idx, agent2_col_idx, agent2_action_idx)

        if (agent1_row_idx, agent1_col_idx) == TARGET1_POS and (agent2_row_idx, agent2_col_idx) == TARGET2_POS:
            global_reward = REWARD_CAPTURE_BOTH
            done_agent = True
        elif (agent1_row_idx, agent1_col_idx) == TARGET1_POS or (agent2_row_idx, agent2_col_idx) == TARGET2_POS:
            global_reward = REWARD_CAPTURE_ONE
            done_agent = True
        else:
            global_reward = -2

        sum_global_reward += global_reward

        old_Q_table_joint = Q_table_joint[agent1_old_row_idx, agent1_old_col_idx, agent2_old_row_idx, agent2_old_col_idx, agent1_action_idx, agent2_action_idx]
        temporal_difference_joint = global_reward + (GAMMA * np.max(Q_table_joint[agent1_row_idx, agent1_col_idx, agent2_row_idx, agent2_col_idx])) \
                                    - Q_table_joint[agent1_old_row_idx, agent1_old_col_idx, agent2_old_row_idx, agent2_old_col_idx, agent1_action_idx, agent2_action_idx]

        new_Q_table_joint = old_Q_table_joint + (ALPHA * temporal_difference_joint)
        Q_table_joint[agent1_old_row_idx, agent1_old_col_idx, agent2_old_row_idx, agent2_old_col_idx, agent1_action_idx, agent2_action_idx] = new_Q_table_joint

    set_global_reward.append(sum_global_reward)

# print("Q_table_joint")
# print(Q_table_joint)

# print(set_global_reward)

