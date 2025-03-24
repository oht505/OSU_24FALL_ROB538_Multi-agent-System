import mas_hw2_singleAgent_singleTarget
import mas_hw2_twoAgents_twoTargets
import mas_hw2_twoAgents_twoTargets_cooperation
import matplotlib.pyplot as plt

set_reward = mas_hw2_singleAgent_singleTarget.set_reward

set_reward_both = mas_hw2_twoAgents_twoTargets.set_reward_both

set_global_rewards = mas_hw2_twoAgents_twoTargets_cooperation.set_global_reward

NUM_EPISODES = mas_hw2_twoAgents_twoTargets_cooperation.NUM_EPISODES

time = []
for t in range(NUM_EPISODES):
    time.append(t)

plt.figure(figsize=(10,6))

plt.plot(time, set_global_rewards, label="Cooperation", color='blue')
plt.plot(time, set_reward_both, label="Non-Cooperation", color='red')
plt.plot(time, set_reward, label="Single Agent", color='green')
plt.title("Accumulative Reward")
plt.xlabel("Number of episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()