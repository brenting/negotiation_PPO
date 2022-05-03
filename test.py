from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from agent.ppo_agent import PPOAgent
from environment.opponents import (
    BoulwareAgent,
    ConcederAgent,
    HardlinerAgent,
    LinearAgent,
    RandomAgent,
    StupidAgent,
)

# collect domains and opponents for testing (don't initialise the opponents)
domains = get_domains("environment/domains/test")
opponents = (
    BoulwareAgent,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)
agent = PPOAgent.load("checkpoint.pkl")

# test on 50 random negotiation sessions and gather average results
rewards = []
opp_rewards = []
welfare = []
agreements = 0
for _ in range(50):
    obs = env.reset(agent)
    done = False
    while not done:
        domain = env.current_domain
        #print(domain[0])
        action = agent.select_action(obs)
        obs, reward, done, opp_reward = env.step(action)
        if done:
            if reward != 0:
                agreements += 1
            rewards.append(reward)

            if opp_reward is None:
                welfare.append(reward)
            else:
                opp_rewards.append(opp_reward)
                welfare.append(reward + opp_reward)
            break

# print results
print(f"Average reward: {sum(rewards)/len(rewards)}")
print(f"Average opponent reward: {sum(opp_rewards)/len(opp_rewards)}")
print(f"Average social welfare: {sum(welfare)/len(welfare)}")
print(f"Percentage of agreements: {agreements * 2}")  # (agreements / 50) * 100 = agreements * 2
# TODO distance to pareto frontier
