import torch
import json

from agent.ppo_agent import PPOAgent
from agent.utils.ppo import DEVICE
from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from environment.opponents import (
    BoulwareAgent,
    ConcederAgent,
    HardlinerAgent,
    LinearAgent,
    RandomAgent,
    StupidAgent,
    CSE3210,
)
from utils.plot_trace import plot_nash_kalai_pareto, plot_trace, distance_to_nash

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


def test(agent):
    # create environment and PPO agent
    env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000, seed=42)

    # test on 50 random negotiation sessions and gather average results
    rewards = []
    opp_rewards = []
    welfare = []
    agreements = 0
    my_prof = None
    switch = False
    nash_avg = []
    for ind in range(50):
        obs = env.reset(agent)
        my_prof = str(env.my_domain)[-13:]
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
            action = agent.select_action(obs, training=False)
            obs, reward, done, opp_reward = env.step(action)
            switch = False
            if done:
                if my_prof == "profileB.json":
                    switch = True
                nash_avg.append(distance_to_nash(env.trace, nash_point, switch))
                try:
                    print(env.trace["actions"][len(env.trace["actions"]) - 1]['Accept']['utilities'],
                          env.current_domain[0])
                    reward = env.trace["actions"][len(env.trace["actions"]) - 1]['Accept']['utilities']['me']
                    opp_reward = env.trace["actions"][len(env.trace["actions"]) - 1]['Accept']['utilities']['opponent']
                except KeyError:
                    print("probs no accept")
                if reward != 0:
                    agreements += 1
                rewards.append(reward)

                if opp_reward is None:
                    welfare.append(reward)
                else:
                    opp_rewards.append(opp_reward)
                    welfare.append(reward + opp_reward)
                break
            # plot_nash_kalai_pareto(env.trace, nash_point, kalai_point, pareto_utilities, "results/" + str(ind) + "evaluation_plot.html", switch)
            # plot_trace(env.trace, "results/" + str(ind) + "trace_plot.html")
    plot_nash_kalai_pareto(env.trace, nash_point, kalai_point, pareto_utilities, "results/evaluation_plot.html", switch)
    plot_trace(env.trace, "results/trace_plot.html")
    # print results
    print(f"Average reward: {sum(rewards) / len(rewards)}")
    print(f"Average opponent reward: {sum(opp_rewards) / len(opp_rewards)}")
    print(f"Average social welfare: {sum(welfare) / len(welfare)}")
    print(f"Percentage of agreements: {agreements * 2}")  # (agreements / 50) * 100 = agreements * 2
    print(f"Distance to Nash Point: {sum(nash_avg) / len(nash_avg)}")


test(PPOAgent.load("checkpoint.pkl"))
