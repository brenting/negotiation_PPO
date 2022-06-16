import json

import matplotlib.pyplot as plt
import numpy as np

from agent.concession_agent import ConcessionAgent
from agent.ppo_agent import PPOAgent
from environment.domains import get_domains
from environment.negotiation import NegotiationEnv
from agent.baseline_agent import BaselineAgent
# collect domains and opponents for testing (don't initialise the opponents)
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


def test():
    global agent, rewards
    domains = get_domains("environment/domains/test")
    opponents = (
        #HardlinerAgent,
        #ConcederAgent,
        #BoulwareAgent,
        #BaselineAgent,
        CSE3210.Agent2,

        CSE3210.Agent18,

         CSE3210.Agent26,

        CSE3210.Agent41,

         CSE3210.Agent52,

       CSE3210.Agent64,
        CSE3210.Agent67,
         CSE3210.Agent68,

       CSE3210.Agent78,

    )
    # create environment and PPO agent
    env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=6000, verbose=True, seed=42)
    agent = ConcessionAgent.load("conceshpoint.pkl")
    #agent = PPOAgent.load("checkpoint.pkl")
    #agent = BaselineAgent()
    # test on 50 random negotiation sessions and gather average results
    rewards = []
    opp_rewards = []
    welfare = []
    agreements = 0
    my_prof = None
    switch = False
    nash_avg = []
    concesh = []
    for _ in range(5):
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

        try:
            f = open("learned_values/" + env.opponent.__class__.__name__ + ".txt", "r")
            agent.opp_concession = float(f.read())
            f.close()
            #concesh.append(agent.opp_concession)
        except FileNotFoundError:
            agent.opp_concession = 0

        while not done:
            action = agent.select_action(obs, training=False)
            concesh.append(agent.opp_concession)
            obs, reward, done, opp_reward = env.step(action)
            #print(agent.opp_concession)
            switch = False
            if done:
                # print(agent.opp_concession)
                if my_prof == "profileB.json":
                    switch = True

                try:
                    nash_avg.append(distance_to_nash(env.trace, nash_point, switch))
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

                # write e value to file
                f = open("learned_values/" + env.opponent.__class__.__name__ + ".txt", "w")
                try:
                    e = agent.opp_concession
                except AttributeError:
                    e = 0.0
                f.write(str(e))
                f.close()

                break
    plot_nash_kalai_pareto(env.trace, nash_point, kalai_point, pareto_utilities, "results/evaluation_plot.html", switch)
    plot_trace(env.trace, "results/trace_plot.html")

    plot_trendline(rewards, "rewards_plot")
    plot_trendline(concesh, "concesh_plot")
    # print results
    print(f"Average reward: {sum(rewards) / len(rewards)} with variance {np.var(rewards)}")
    print(f"Average opponent reward: {sum(opp_rewards) / len(opp_rewards)}")
    print(f"Average social welfare: {sum(welfare) / len(welfare)}")
    print(f"Percentage of agreements: {agreements / len(rewards)}")
    print(f"Distance to Nash Point: {sum(nash_avg) / len(nash_avg)}")


def plot_trendline(data, filename):
    x, y = np.arange(len(data)), data
    # create scatterplot
    plt.scatter(x, y)
    plt.xlabel("Total rounds")
    plt.ylabel("Estimated opponent concession parameter")
    # calculate equation for trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    # add trendline to plot
    plt.plot(x, p(x))
    # plt.plot(np.arange(len(rewards)), rewards)
    plt.savefig("results/" + filename)
    plt.close()


test()
