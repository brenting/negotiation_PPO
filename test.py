from agent.ppo_agent import PPOAgent
from environment.domains import get_domains
from environment.negotiation import NegotiationEnv
from environment.opponents import (
    CSE3210,
)

# collect domains and opponents for testing (don't initialise the opponents)
domains = get_domains("environment/domains/test")
opponents = (
    # ConcederAgent,
    # HardlinerAgent,
    # BoulwareAgent,
    # LinearAgent,
    # RandomAgent,
    # StupidAgent,
    CSE3210.Agent2,
    # CSE3210.Agent3,
    # CSE3210.Agent7,
    # CSE3210.Agent11,
    # CSE3210.Agent14,
    CSE3210.Agent18,
    # CSE3210.Agent19,
    # CSE3210.Agent22,
    # CSE3210.Agent24,
    # CSE3210.Agent25,
    CSE3210.Agent26,
    # CSE3210.Agent27,
    # CSE3210.Agent29,
    # CSE3210.Agent32,
    # CSE3210.Agent33,
    CSE3210.Agent41,
    CSE3210.Agent43,
    # CSE3210.Agent50,
    CSE3210.Agent52,
    # CSE3210.Agent55,
    # CSE3210.Agent58,
    # CSE3210.Agent61,
    CSE3210.Agent64,
    CSE3210.Agent67,
    CSE3210.Agent68,
    # CSE3210.Agent70,
    CSE3210.Agent78,
)


# test on 50 random negotiation sessions and gather average results
def run_test_negotiations(agent):
    rewards = []
    opp_rewards = []
    for _ in range(50):
        obs = env.reset(agent)
        done = False
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, done, opp_reward = env.step(action)
            if done:
                rewards.append(reward)
                opp_rewards.append(opp_reward)
                break
    return rewards, opp_rewards


def collect_averages(agents):
    agents_rewards = []
    agents_opp_rewards = []
    for agent in agents:
        result = run_test_negotiations(agent)
        agents_rewards = agents_rewards + result[0]
        agents_opp_rewards = agents_opp_rewards + result[1]
    return agents_rewards, agents_opp_rewards


# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000, seed=42)
range_5 = range(1, 6)

# # Base PPO agent with state using 3 previous offers and progress
# base_agents = [PPOAgent.load("checkpoints/checkpoint_base_" + str(x) + ".pkl") for x in range_5]
# base_agent_rewards, base_agent_opp_rewards = collect_averages(base_agents)
#
# with open('base_agent_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in base_agent_rewards)
#
# with open('base_agent_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in base_agent_opp_rewards)
#
# # Base PPO agent but now with 5 previous offers
# base_agents_5_offers = [PPOAgent.load("checkpoints/checkpoint_base_5_" + str(x) + ".pkl") for x in range_5]
# base_agent_5_rewards, base_agent_5_opp_rewards = collect_averages(base_agents_5_offers)
#
# with open('base_agent_5_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in base_agent_5_rewards)
#
# with open('base_agent_5_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in base_agent_5_opp_rewards)
#
# # Base PPO agent but now with 50 previous offers
# base_agents_50_offers = [PPOAgent.load("checkpoints/checkpoint_base_50_" + str(x) + ".pkl") for x in range_5]
# base_agent_50_rewards, base_agent_50_opp_rewards = collect_averages(base_agents_50_offers)
#
# with open('base_agent_50_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in base_agent_50_rewards)
#
# with open('base_agent_50_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in base_agent_50_opp_rewards)
#
# mean
# stats_agents_1 = [PPOAgent.load("checkpoints/checkpoint_stats_1_" + str(x) + ".pkl") for x in range_5]
# stats_agent_1_rewards, stats_agent_1_opp_rewards = collect_averages(stats_agents_1)
#
# with open('stats_agent_1_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_1_rewards)
#
# with open('stats_agent_1_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_1_opp_rewards)
#
# mean + number_of_bids
# stats_agents_2 = [PPOAgent.load("checkpoints/checkpoint_stats_2_" + str(x) + ".pkl") for x in range_5]
# stats_agent_2_rewards, stats_agent_2_opp_rewards = collect_averages(stats_agents_2)
#
# with open('stats_agent_2_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_2_rewards)
#
# with open('stats_agent_2_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_2_opp_rewards)
#
# # mean + number_of_bids + own_std
# stats_agents_3 = [PPOAgent.load("checkpoints/checkpoint_stats_3_" + str(x) + ".pkl") for x in range_5]
# stats_agent_3_rewards, stats_agent_3_opp_rewards = collect_averages(stats_agents_3)
#
# with open('stats_agent_3_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_3_rewards)
#
# with open('stats_agent_3_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_3_opp_rewards)
#
# # mean + number_of_bids + own_std + own_median
# stats_agents_4 = [PPOAgent.load("checkpoints/checkpoint_stats_4_" + str(x) + ".pkl") for x in range_5]
# stats_agent_4_rewards, stats_agent_4_opp_rewards = collect_averages(stats_agents_4)
#
# with open('stats_agent_4_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_4_rewards)
#
# with open('stats_agent_4_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_4_opp_rewards)
#
# # mean + number_of_bids + own_std + own_median + own_mode
# stats_agents_5 = [PPOAgent.load("checkpoints/checkpoint_stats_5_" + str(x) + ".pkl") for x in range_5]
# stats_agent_5_rewards, stats_agent_5_opp_rewards = collect_averages(stats_agents_5)
#
# with open('stats_agent_5_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_5_rewards)
#
# with open('stats_agent_5_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_5_opp_rewards)
#
# mean + number_of_bids + own_std + own_median + own_mode + own_range
# stats_agents_6 = [PPOAgent.load("checkpoints/checkpoint_stats_6_" + str(x) + ".pkl") for x in range_5]
# stats_agent_6_rewards, stats_agent_6_opp_rewards = collect_averages(stats_agents_6)
#
# with open('stats_agent_6_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_6_rewards)
#
# with open('stats_agent_6_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_6_opp_rewards)
#
# mean + number_of_bids + own_std + own_median + own_mode + own_range + corr
# stats_agents_7 = [PPOAgent.load("checkpoints/checkpoint_stats_7_" + str(x) + ".pkl") for x in range_5]
# stats_agent_7_rewards, stats_agent_7_opp_rewards = collect_averages(stats_agents_7)
#
# with open('stats_agent_7_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_7_rewards)
#
# with open('stats_agent_7_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_7_opp_rewards)
#
# mean + number_of_bids + own_std + own_median + own_mode + own_range + corr + opp_mean
stats_agents_8 = [PPOAgent.load("checkpoints/checkpoint_stats_8_" + str(x) + ".pkl") for x in range_5]
stats_agent_8_rewards, stats_agent_8_opp_rewards = collect_averages(stats_agents_8)

with open('stats_agent_8_rewards.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % reward for reward in stats_agent_8_rewards)

with open('stats_agent_8_opp_rewards.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % reward for reward in stats_agent_8_opp_rewards)

# # mean + number_of_bids + own_std + own_median + own_mode + own_range + corr + opp_mean + opp_std
# stats_agents_9 = [PPOAgent.load("checkpoints/checkpoint_stats_9_" + str(x) + ".pkl") for x in range_5]
# stats_agent_9_rewards, stats_agent_9_opp_rewards = collect_averages(stats_agents_9)
#
# with open('stats_agent_9_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_9_rewards)
#
# with open('stats_agent_9_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_9_opp_rewards)
#
# # mean + number_of_bids + own_std + own_median + own_mode + own_range + corr + opp_mean + opp_std + opp_median
# stats_agents_10 = [PPOAgent.load("checkpoints/checkpoint_stats_10_" + str(x) + ".pkl") for x in range_5]
# stats_agent_10_rewards, stats_agent_10_opp_rewards = collect_averages(stats_agents_10)
#
# with open('stats_agent_10_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_10_rewards)
#
# with open('stats_agent_10_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_10_opp_rewards)
#
# # mean + number_of_bids + own_std + own_median + own_mode + own_range + corr + opp_mean + opp_std + opp_median + opp_mode
# stats_agents_11 = [PPOAgent.load("checkpoints/checkpoint_stats_11_" + str(x) + ".pkl") for x in range_5]
# stats_agent_11_rewards, stats_agent_11_opp_rewards = collect_averages(stats_agents_11)
#
# with open('stats_agent_11_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_11_rewards)
#
# with open('stats_agent_11_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_11_opp_rewards)
#
# # mean + number_of_bids + own_std + own_median + own_mode + own_range + corr + opp_mean + opp_std + opp_median + opp_mode + opp_range
# stats_agents_12 = [PPOAgent.load("checkpoints/checkpoint_stats_12_" + str(x) + ".pkl") for x in range_5]
# stats_agent_12_rewards, stats_agent_12_opp_rewards = collect_averages(stats_agents_12)
#
# with open('stats_agent_12_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_12_rewards)
#
# with open('stats_agent_12_opp_rewards.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % reward for reward in stats_agent_12_opp_rewards)

# # print results
# print(f"Average reward: {sum(base_agent_rewards) / len(base_agent_rewards)}")
# print(f"Average opponent reward: {sum(base_agent_opp_rewards) / len(base_agent_opp_rewards)}")
