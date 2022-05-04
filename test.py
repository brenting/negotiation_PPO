import json

from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from agent.ppo_agent import PPOAgent
from agent.TraderAgent import TraderAgent
from agent.acceptance_agent import AcceptanceAgent
from environment.opponents import (
    BoulwareAgent,
    ConcederAgent,
    HardlinerAgent,
    LinearAgent,
    #RandomAgent,
    #StupidAgent,
)

# collect domains and opponents for testing (don't initialise the opponents)
from utils.plot_trace import plot_nash_kalai_pareto, plot_trace

domains = get_domains("environment/domains/test")
opponents = (
    BoulwareAgent,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)
agent = AcceptanceAgent.load("checkpoint.pkl")

# test on 50 random negotiation sessions and gather average results
rewards = []
opp_rewards = []
welfare = []
agreements = 0
for _ in range(50):
    obs = env.reset(agent)
    done = False
    domain = env.current_domain
    path = str(domain[0].getPath())
    path = path.replace("profileA", "specials", 1)
    path = path.replace("profileB", "specials", 1)
    # Get nash, kalai, pareto points from specials.json
    file = open(path)
    json_contents = json.load(file)
    nash_point = json_contents.get('nash').get('utility')
    kalai_point = json_contents.get('kalai').get('utility')
    pareto_front = json_contents.get('pareto_front')
    pareto_utilities = []
    for bid in pareto_front:
        pareto_utilities.append(bid.get('utility'))

    while not done:
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
    plot_nash_kalai_pareto(env.trace, nash_point, kalai_point, pareto_utilities, "evaluation_plot.html")
    plot_trace(env.trace, "trace_plot.html")

# print results
print(f"Average reward: {sum(rewards)/len(rewards)}")
print(f"Average opponent reward: {sum(opp_rewards)/len(opp_rewards)}")
print(f"Average social welfare: {sum(welfare)/len(welfare)}")
print(f"Percentage of agreements: {agreements * 2}")  # (agreements / 50) * 100 = agreements * 2
# TODO distance to pareto frontier
