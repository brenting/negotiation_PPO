import numpy as np
from matplotlib import pyplot as plt

with open('base_agent_rewards.txt', 'r') as filehandle:
    base_agent_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('base_agent_opp_rewards.txt', 'r') as filehandle:
    base_agent_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_base_agent_rewards = np.sum(np.array([base_agent_rewards, base_agent_opp_rewards]), 0)

with open('base_agent_5_rewards.txt', 'r') as filehandle:
    base_agent_5_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('base_agent_5_opp_rewards.txt', 'r') as filehandle:
    base_agent_5_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_base_agent_5_rewards = np.sum(np.array([base_agent_5_rewards, base_agent_5_opp_rewards]), 0)

with open('base_agent_50_rewards.txt', 'r') as filehandle:
    base_agent_50_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('base_agent_50_opp_rewards.txt', 'r') as filehandle:
    base_agent_50_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_base_agent_50_rewards = np.sum(np.array([base_agent_50_rewards, base_agent_50_opp_rewards]), 0)

with open('stats_agent_1_rewards.txt', 'r') as filehandle:
    stats_agent_1_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_1_opp_rewards.txt', 'r') as filehandle:
    stats_agent_1_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_1_rewards = np.sum(np.array([stats_agent_1_rewards, stats_agent_1_opp_rewards]), 0)

with open('stats_agent_2_rewards.txt', 'r') as filehandle:
    stats_agent_2_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_2_opp_rewards.txt', 'r') as filehandle:
    stats_agent_2_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_2_rewards = np.sum(np.array([stats_agent_2_rewards, stats_agent_2_opp_rewards]), 0)

with open('stats_agent_3_rewards.txt', 'r') as filehandle:
    stats_agent_3_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_3_opp_rewards.txt', 'r') as filehandle:
    stats_agent_3_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_3_rewards = np.sum(np.array([stats_agent_3_rewards, stats_agent_3_opp_rewards]), 0)

with open('stats_agent_4_rewards.txt', 'r') as filehandle:
    stats_agent_4_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_4_opp_rewards.txt', 'r') as filehandle:
    stats_agent_4_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_4_rewards = np.sum(np.array([stats_agent_4_rewards, stats_agent_4_opp_rewards]), 0)

with open('stats_agent_5_rewards.txt', 'r') as filehandle:
    stats_agent_5_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_5_opp_rewards.txt', 'r') as filehandle:
    stats_agent_5_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_5_rewards = np.sum(np.array([stats_agent_5_rewards, stats_agent_5_opp_rewards]), 0)

with open('stats_agent_6_rewards.txt', 'r') as filehandle:
    stats_agent_6_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_6_opp_rewards.txt', 'r') as filehandle:
    stats_agent_6_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_6_rewards = np.sum(np.array([stats_agent_6_rewards, stats_agent_6_opp_rewards]), 0)

with open('stats_agent_7_rewards.txt', 'r') as filehandle:
    stats_agent_7_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_7_opp_rewards.txt', 'r') as filehandle:
    stats_agent_7_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_7_rewards = np.sum(np.array([stats_agent_7_rewards, stats_agent_7_opp_rewards]), 0)

with open('stats_agent_8_rewards.txt', 'r') as filehandle:
    stats_agent_8_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_8_opp_rewards.txt', 'r') as filehandle:
    stats_agent_8_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_8_rewards = np.sum(np.array([stats_agent_8_rewards, stats_agent_8_opp_rewards]), 0)

with open('stats_agent_9_rewards.txt', 'r') as filehandle:
    stats_agent_9_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_9_opp_rewards.txt', 'r') as filehandle:
    stats_agent_9_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_9_rewards = np.sum(np.array([stats_agent_9_rewards, stats_agent_9_opp_rewards]), 0)

with open('stats_agent_10_rewards.txt', 'r') as filehandle:
    stats_agent_10_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_10_opp_rewards.txt', 'r') as filehandle:
    stats_agent_10_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_10_rewards = np.sum(np.array([stats_agent_10_rewards, stats_agent_10_opp_rewards]), 0)

with open('stats_agent_11_rewards.txt', 'r') as filehandle:
    stats_agent_11_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_11_opp_rewards.txt', 'r') as filehandle:
    stats_agent_11_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_11_rewards = np.sum(np.array([stats_agent_11_rewards, stats_agent_11_opp_rewards]), 0)

with open('stats_agent_12_rewards.txt', 'r') as filehandle:
    stats_agent_12_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

with open('stats_agent_12_opp_rewards.txt', 'r') as filehandle:
    stats_agent_12_opp_rewards = [float(reward.rstrip()) for reward in filehandle.readlines()]

social_stats_agent_12_rewards = np.sum(np.array([stats_agent_12_rewards, stats_agent_12_opp_rewards]), 0)

# ABLATION STUDY
agents = ['B', 'B+5', 'B+50', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
x = np.arange(len(agents))
width = 0.35
average_rewards = [np.mean(base_agent_rewards), np.mean(base_agent_5_rewards), np.mean(base_agent_50_rewards),
                   np.mean(stats_agent_1_rewards), np.mean(stats_agent_2_rewards), np.mean(stats_agent_3_rewards),
                   np.mean(stats_agent_4_rewards), np.mean(stats_agent_5_rewards), np.mean(stats_agent_6_rewards),
                   np.mean(stats_agent_7_rewards), np.mean(stats_agent_8_rewards), np.mean(stats_agent_9_rewards),
                   np.mean(stats_agent_10_rewards), np.mean(stats_agent_11_rewards), np.mean(stats_agent_12_rewards)]
average_opp_rewards = [np.mean(base_agent_opp_rewards), np.mean(base_agent_5_opp_rewards),
                       np.mean(base_agent_50_opp_rewards), np.mean(stats_agent_1_opp_rewards),
                       np.mean(stats_agent_2_opp_rewards), np.mean(stats_agent_3_opp_rewards),
                       np.mean(stats_agent_4_opp_rewards), np.mean(stats_agent_5_opp_rewards),
                       np.mean(stats_agent_6_opp_rewards), np.mean(stats_agent_7_opp_rewards),
                       np.mean(stats_agent_8_opp_rewards), np.mean(stats_agent_9_opp_rewards),
                       np.mean(stats_agent_10_opp_rewards), np.mean(stats_agent_11_opp_rewards),
                       np.mean(stats_agent_12_opp_rewards)]
rewards_error = [np.std(base_agent_rewards), np.std(base_agent_5_rewards), np.std(base_agent_50_rewards),
                 np.std(stats_agent_1_rewards), np.std(stats_agent_2_rewards), np.std(stats_agent_3_rewards),
                 np.std(stats_agent_4_rewards), np.std(stats_agent_5_rewards), np.std(stats_agent_6_rewards),
                 np.std(stats_agent_7_rewards), np.std(stats_agent_8_rewards), np.std(stats_agent_9_rewards),
                 np.std(stats_agent_10_rewards), np.std(stats_agent_11_rewards), np.std(stats_agent_12_rewards)]
rewards_opp_error = [np.std(base_agent_opp_rewards), np.std(base_agent_5_opp_rewards),
                     np.std(base_agent_50_opp_rewards), np.std(stats_agent_1_opp_rewards),
                     np.std(stats_agent_2_opp_rewards), np.std(stats_agent_3_opp_rewards),
                     np.std(stats_agent_4_opp_rewards), np.std(stats_agent_5_opp_rewards),
                     np.std(stats_agent_6_opp_rewards), np.std(stats_agent_7_opp_rewards),
                     np.std(stats_agent_8_opp_rewards), np.std(stats_agent_9_opp_rewards),
                     np.std(stats_agent_10_opp_rewards), np.std(stats_agent_11_opp_rewards),
                     np.std(stats_agent_12_opp_rewards)]

fig, ax = plt.subplots(figsize=(8, 5))
ar = ax.bar(x - width / 2, average_rewards, width, yerr=rewards_error, align='center', capsize=3, label='u')
ae = ax.bar(x + width / 2, average_opp_rewards, width, yerr=rewards_opp_error, align='center', capsize=3, label='u_opp')
ax.set_ylabel('Utility')
ax.set_xlabel('Agents')
ax.set_title('Average utility for the agent versions in an ablation study')
ax.set_xticks(x, agents)

# Save the figure and show
plt.legend()
plt.tight_layout()
plt.savefig('Ablation of all agents')
plt.show()

# SOCIAL WELFARE
agents = ['B', 'B+5', 'B+50', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
x = np.arange(len(agents))

social_average_rewards = [np.mean(social_base_agent_rewards), np.mean(social_base_agent_5_rewards),
                          np.mean(social_base_agent_50_rewards), np.mean(social_stats_agent_1_rewards),
                          np.mean(social_stats_agent_2_rewards), np.mean(social_stats_agent_3_rewards),
                          np.mean(social_stats_agent_4_rewards), np.mean(social_stats_agent_5_rewards),
                          np.mean(social_stats_agent_6_rewards), np.mean(social_stats_agent_7_rewards),
                          np.mean(social_stats_agent_8_rewards), np.mean(social_stats_agent_9_rewards),
                          np.mean(social_stats_agent_10_rewards), np.mean(social_stats_agent_11_rewards),
                          np.mean(social_stats_agent_12_rewards)]
social_rewards_error = [np.std(social_base_agent_rewards), np.std(social_base_agent_5_rewards),
                        np.std(social_base_agent_50_rewards), np.std(social_stats_agent_1_rewards),
                        np.std(social_stats_agent_2_rewards), np.std(social_stats_agent_3_rewards),
                        np.std(social_stats_agent_4_rewards), np.std(social_stats_agent_5_rewards),
                        np.std(social_stats_agent_6_rewards), np.std(social_stats_agent_7_rewards),
                        np.std(social_stats_agent_8_rewards), np.std(social_stats_agent_9_rewards),
                        np.std(social_stats_agent_10_rewards), np.std(social_stats_agent_11_rewards),
                        np.std(social_stats_agent_12_rewards)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x, social_average_rewards, yerr=social_rewards_error, align='center', capsize=3, label='u_social')
ax.set_ylabel('Utility')
ax.set_xlabel('Agents')
ax.set_title('Average simple social welfare utility for the agent versions in an ablation study')
ax.set_xticks(x, agents)

# Save the figure and show
plt.tight_layout()
plt.savefig('Social welfare ablation of all agents')
plt.show()

print('ar:', [round(reward, 3) for reward in average_rewards])
print('as:', [round(reward, 3) for reward in rewards_error])
print('or:', [round(reward, 3) for reward in average_opp_rewards])
print('os:', [round(reward, 3) for reward in rewards_opp_error])
print('sr:', [round(reward, 3) for reward in social_average_rewards])
print('ss:', [round(reward, 3) for reward in social_rewards_error])
