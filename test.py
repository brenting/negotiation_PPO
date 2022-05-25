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
    # SelfPlayAgent
)

from utils.plot_trace import plot_nash_kalai_pareto, plot_trace, distance_to_nash

from environment.opponents.CSE3210 import (
    Agent2,
    Agent3,
    Agent7,
    Agent11,
    Agent14,
    Agent18,
    Agent19,
    Agent22,
    Agent24,
    Agent25,
    Agent26,
    Agent27,
    Agent29,
    Agent32,
    Agent33,
    Agent41,
    Agent43,
    Agent50,
    Agent52,
    Agent55,
    Agent58,
    Agent61,
    Agent64,
    Agent67,
    Agent68,
    Agent70,
    Agent78,
)


def test(agent):
    # agent.ppo.policy_old.action_var = torch.full((2,), 0.01).to(DEVICE)
    # collect domains and opponents for testing (don't initialise the opponents)--
    domains = get_domains("environment/domains/test")
    opponents = (
        Agent2,
        Agent3,
        Agent7,
        Agent11,
        Agent14,
        Agent18,
        Agent19,
        Agent22,
        Agent24,
        Agent25,
        Agent26,
        Agent27,
        Agent29,
        Agent32,
        Agent33,
        Agent41,
        Agent43,
        Agent50,
        Agent52,
        Agent55,
        Agent58,
        Agent61,
        Agent64,
        Agent67,
        Agent68,
        Agent70,
        Agent78,
        # BoulwareAgent,
        # ConcederAgent,
        # HardlinerAgent,
        # LinearAgent,
        # RandomAgent,
        # StupidAgent,
        # SelfPlayAgent
    )

    # create environment and PPO agent
    env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)

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
            action = agent.select_action(obs)
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
