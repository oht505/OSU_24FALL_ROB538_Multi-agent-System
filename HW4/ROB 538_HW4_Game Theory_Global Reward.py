import numpy as np
import pandas as pd

# Parameters
num_agents = 50
k = 6
b = 4

# Local Reward Function
def local_reward_function(x_k, b):
    return x_k * np.exp(-x_k / b)

# Initialize agents to random nights
agents_chosen_night = np.random.choice(range(1, k + 1), num_agents)
print(f'agents_chosen_night: {agents_chosen_night}')

# Function to calculate local_rewards for all nights
def calculate_all_local_rewards(agents_chosen_night, k, b):
    agents_per_night = [np.sum(agents_chosen_night == night) for night in range(1, k + 1)]
    local_rewards = [local_reward_function(x, b) for x in agents_per_night]
    return local_rewards, agents_per_night


# Nash Equilibrium Simulation
nash_equilibrium_reached = False

while (not nash_equilibrium_reached):
    nash_equilibrium_reached = True

    # Iterate through each agent to evaluate their strategy
    for agent in range(num_agents):
        chosen_night = agents_chosen_night[agent]

        # Calculate current local reward
        local_rewards, agents_per_night = calculate_all_local_rewards(agents_chosen_night, k, b)
        current_global_reward = np.sum(local_rewards)

        # Check if moving to a different night provides a better local reward
        better_night_found = False
        for new_night in range(1, k + 1):
            if new_night != chosen_night:
                agents_chosen_night[agent] = new_night
                new_local_rewards, _ = calculate_all_local_rewards(agents_chosen_night, k, b)
                new_global_reward = np.sum(new_local_rewards)

                # Compare new global reward with the current global reward
                if new_global_reward > current_global_reward:
                    better_night_found = True
                    nash_equilibrium_reached = False
                    break

        # If no better night found, go back to the original night
        if not better_night_found:
            agents_chosen_night[agent] = chosen_night

# Final Results
final_local_rewards, final_agents_per_night = calculate_all_local_rewards(agents_chosen_night, k, b)
final_global_reward = np.sum(final_local_rewards)
print(f'Final_local_rewards: {final_local_rewards}')
print(f'Final_global_reward: {final_global_reward}')
print(f'Final_agents_per_night: {final_agents_per_night}')
print()

# Show Results
nash_equilibrium_df = pd.DataFrame({
    "Night": [f"Night {i + 1}" for i in range(k)],
    "Number of Agents Attending": final_agents_per_night,
    "Global_reward": final_global_reward
})

print(nash_equilibrium_df)
