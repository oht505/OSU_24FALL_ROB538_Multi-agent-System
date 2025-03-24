import numpy as np
import random

# Gridworld parameters
GRID_HEIGHT = 5
GRID_WIDTH = 10
TARGET1_POS = (3, 7)
TARGET2_POS = (2, 9)
AGENT1_START = (0, 0)
AGENT2_START = (0, 0)
ACTIONS = ['up', 'down', 'left', 'right']
REWARD_CAPTURE_BOTH = 40
REWARD_CAPTURE_ONE = 20
REWARD_STEP = -1

# Q-learning parameters
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
NUM_EPISODES = 1000

# Initialize a joint Q-table
Q_table_joint = np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_HEIGHT, GRID_WIDTH, len(ACTIONS), len(ACTIONS)))

def get_action_index(action):
    return ACTIONS.index(action)

def choose_joint_action(state):
    x1, y1, x2, y2 = state
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS), random.choice(ACTIONS)
    else:
        action_idx = np.unravel_index(np.argmax(Q_table_joint[x1, y1, x2, y2]), (len(ACTIONS), len(ACTIONS)))
        return ACTIONS[action_idx[0]], ACTIONS[action_idx[1]]

def step(state, action):
    x, y = state
    if action == 'up':
        x = max(x - 1, 0)
    elif action == 'down':
        x = min(x + 1, GRID_HEIGHT - 1)
    elif action == 'left':
        y = max(y - 1, 0)
    elif action == 'right':
        y = min(y + 1, GRID_WIDTH - 1)
    return (x, y)

# Joint Q-learning algorithm
for episode in range(NUM_EPISODES):
    state_agent1 = AGENT1_START
    state_agent2 = AGENT2_START
    done = False

    while not done:
        state = (*state_agent1, *state_agent2)
        #print(state)
        action_agent1, action_agent2 = choose_joint_action(state)
        new_state_agent1 = step(state_agent1, action_agent1)
        new_state_agent2 = step(state_agent2, action_agent2)

        # Determine global reward
        if new_state_agent1 == TARGET1_POS and new_state_agent2 == TARGET2_POS:
            reward = REWARD_CAPTURE_BOTH
            done = True
        elif new_state_agent1 == TARGET1_POS or new_state_agent2 == TARGET2_POS:
            reward = REWARD_CAPTURE_ONE
            done = True
        else:
            reward = REWARD_STEP

        # Update Q-table
        x1, y1 = state_agent1
        x2, y2 = state_agent2
        a1_idx = get_action_index(action_agent1)
        a2_idx = get_action_index(action_agent2)
        nx1, ny1 = new_state_agent1
        nx2, ny2 = new_state_agent2

        Q_table_joint[x1, y1, x2, y2, a1_idx, a2_idx] += ALPHA * (
            reward + GAMMA * np.max(Q_table_joint[nx1, ny1, nx2, ny2]) - Q_table_joint[x1, y1, x2, y2, a1_idx, a2_idx]
        )

        state_agent1 = new_state_agent1
        state_agent2 = new_state_agent2

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

print(Q_table_joint)