import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_AGENTS_CASE_A = 25
NUM_AGENTS_CASE_B = 40
B_CASE_A = 5
B_CASE_B = 4
K_CASE_A = 7
K_CASE_B = 6
TIME_LIMIT = 100

EPSILON = 0.1
ALPHA = 0.9

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
def local_reward_function(xk, b):
    return xk * np.exp(-xk / b)

def system_reward_function(xk, b):
    return np.sum(local_reward_function(xk, b))

# Agent Class
class Agent:
    def __init__(self, k):
        self.k = k
        self.q_action_values = np.zeros(k)

    def choose_action(self):
        if np.random.rand() < EPSILON:
            return np.random.randint(self.k)
        else:
            # np.argmax function always returns 0 when elements have same value
            return np.argmax(self.q_action_values)

            #return my_argmax(self.q_action_values)

    # Simple value update
    def update_q_value(self, night, reward):
        self.q_action_values[night] += ALPHA * (reward - self.q_action_values[night])


# Environment Simulation
def simulate(num_agents, b, k):
    agents = [Agent(k) for _ in range(num_agents)]
    system_rewards = []
    local_rewards = []
    difference_rewards = []
    attendances = []
    accum_local_reward, accum_local_rewards = 0., []
    accum_difference_reward, accum_difference_rewards = 0., []
    accum_system_reward, accum_system_rewards = 0., []

    for week in range(TIME_LIMIT):

        # Initialize attendance
        attendance = np.zeros(k)

        # Each agent selects an action (night to attend)
        actions = [agent.choose_action() for agent in agents]

        # Count attendance for each night
        for action in actions:
            attendance[action] += 1

        # Record attendances for histogram
        attendances.append(attendance)

        # Calculate system and local rewards
        weekly_system_reward = system_reward_function(attendance, b)
        weekly_local_rewards = local_reward_function(attendance, b)

        # Initialize sum of local and difference rewards to zero
        sum_local_rewards = 0
        sum_difference_rewards = 0

        # Compute each agent's rewards
        for i, agent in enumerate(agents):
            chosen_night = actions[i]
            local_reward = weekly_local_rewards[chosen_night]
            counterfactual_attendance = np.copy(attendance)
            counterfactual_attendance[chosen_night] -= 1
            difference_reward = weekly_system_reward - system_reward_function(counterfactual_attendance, b)

            # Update agents' Q-values based on rewards
            agent.update_q_value(chosen_night, difference_reward)

            # Accumulate local and difference rewards
            sum_local_rewards += local_reward
            sum_difference_rewards += difference_reward

        # Mean of local and difference rewards for weekly rewards
        n = len(agents)
        system_rewards.append(weekly_system_reward)
        difference_rewards.append(sum_difference_rewards)
        local_rewards.append(sum_local_rewards)

        # Accumulate three different rewards
        accum_local_reward += sum_local_rewards
        accum_difference_reward += sum_difference_rewards
        accum_system_reward += weekly_system_reward

        accum_local_rewards.append(accum_local_reward)
        accum_difference_rewards.append(accum_difference_reward)
        accum_system_rewards.append(accum_system_reward)

    # Plot Accumulative three different rewards
    plot_three_data(accum_system_rewards, accum_difference_rewards, accum_local_rewards,
                    'Weeks', 'Accumulative Rewards', 'Accumulative',
                    num_agents, b, k)

    # Plot Average of three different rewards
    plot_three_data((np.array(system_rewards)), (np.array(difference_rewards)/n), (np.array(local_rewards)/n),
                    'Weeks', 'Rewards', 'Weekly Average',
                    num_agents, b, k
                    )

    # Plot a histogram of a sample attendance profile
    plot_histogram(range(1, k + 1), attendances[5], num_agents)
    plot_histogram(range(1, k + 1), attendances[-1], num_agents)


simulate(NUM_AGENTS_CASE_A, B_CASE_A, K_CASE_A)
simulate(NUM_AGENTS_CASE_B, B_CASE_B, K_CASE_B)



