import json

import matplotlib.pyplot as plt
import numpy as np

from agent.concession_agent import ConcessionAgent
from environment.domains import get_domains
from environment.negotiation import NegotiationEnv
# collect domains and opponents for testing (don't initialise the opponents)
from environment.opponents.CSE3210 import Agent2, Agent7, Agent11, Agent3, Agent18, Agent19, Agent22, Agent24, Agent25,Agent26, Agent27
from environment.opponents.CSE3210.agent14.agent14 import Agent14
from environment.opponents.boulware_agent.boulware_agent import BoulwareAgent
from utils.plot_trace import plot_nash_kalai_pareto, plot_trace, distance_to_nash


def test():
    global agent, rewards
    domains = get_domains("environment/domains/single")
    opponents = (
        # Agent2,
         #Agent7,
         #Agent11,
        #  Agent3,
        #  Agent14,
        #  Agent18,
        # Agent19,
         #Agent22,
         #Agent24,
         #Agent25,
         Agent26,
         #Agent27
        #BoulwareAgent,
        # HardlinerAgent,
        # ConcederAgent,
        # LinearAgent

    )
    # TODO add more opponents
    # create environment and PPO agent
    env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)
    agent = ConcessionAgent.load("checkpoint.pkl", True)
    # test on 50 random negotiation sessions and gather average results
    rewards = []
    opp_rewards = []
    welfare = []
    agreements = 0
    my_prof = None
    switch = False
    nash_avg = []
    for _ in range(50):
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
        except FileNotFoundError:
            agent.opp_concession = 0

        while not done:
            action = agent.select_action(obs)
            obs, reward, done, opp_reward = env.step(action)
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

    x, y = np.arange(len(rewards)), rewards
    # create scatterplot
    plt.scatter(x, y)

    # calculate equation for trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    # add trendline to plot
    plt.plot(x, p(x))
    #plt.plot(np.arange(len(rewards)), rewards)
    plt.savefig("rewards_plot")
    # print results
    print(f"Average reward: {sum(rewards) / len(rewards)}")
    print(f"Average opponent reward: {sum(opp_rewards) / len(opp_rewards)}")
    print(f"Average social welfare: {sum(welfare) / len(welfare)}")
    print(f"Percentage of agreements: {agreements * 2}")  # (agreements / 50) * 100 = agreements * 2
    print(f"Distance to Nash Point: {sum(nash_avg) / len(nash_avg)}")


test()
# TODO distance to pareto frontier
# TODO Add more plots - example average utility against  1 agent (maybe  for specific domains)

# TODO weaker domains : 17, 10, 14, 3
