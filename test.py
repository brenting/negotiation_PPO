from decimal import Decimal
from math import sqrt
from numpy import average
from agent import ppo_agent
from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from geniusweb.bidspace import AllBidsList
from geniusweb.issuevalue.Domain import Domain
from agent.ppo_agent import PPOAgent
import matplotlib.pyplot as plt
import numpy as np
import timeit
from environment.opponents import (
    BoulwareAgent,
    ConcederAgent,
    HardlinerAgent,
    LinearAgent,
    RandomAgent,
    StupidAgent,
)






def pearsonCorrelationOfBids(domain: Domain, real_utility, predicted_utility):
    #print(actual_utility.getDomain)
    #print(predicted_utility.domain)
    start = timeit.default_timer()
    bids = AllBidsList.AllBidsList(domain)
    #print("Number of bids in bid space: " + str(bids.size()))
    sum_real_utility = 0.0
    sum_predicted_utility = 0.0
    for i in range(bids.size()):
        sum_real_utility += float(real_utility(bids.get(i)))
        sum_predicted_utility += predicted_utility(bids.get(i))
    average_real_utility = sum_real_utility / bids.size()
    average_predicted_utility = sum_predicted_utility / bids.size()

    #this is the top component of the pearson coefficient
    sum_of_products = 0.0
    #this are the bottom components of the pearson coefficient
    realVar = 0.0
    predictedVar = 0.0
    for i in range(bids.size()):
        sum_of_products += (float(real_utility(bids.get(i)))-average_real_utility)*(predicted_utility(bids.get(i))-average_predicted_utility)
        realVar += (float(real_utility(bids.get(i)))-average_real_utility)**2
        predictedVar += (predicted_utility(bids.get(i))-average_predicted_utility)**2
    
    pearsonCorrelation = sum_of_products / sqrt(realVar * predictedVar)
    stop = timeit.default_timer()
    print('Time to calculate pearson correlation: ', stop - start)  

    return pearsonCorrelation
    

# collect domains and opponents for testing (don't initialise the opponents)
domains = get_domains("environment/domains/test")
opponents = (
    BoulwareAgent,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000000)
agent = PPOAgent.load("checkpoint.pkl")

# test on 50 random negotiation sessions and gather average results
rewards = []
opp_rewards = []
for _ in range(50):
    obs = env.reset(agent)
    done = False
    step = 0
    sampleFrequency = 1
    timeEvolutionPearson = []
    while not done:
        action = agent.select_action(obs)
        obs, reward, done, opp_reward = env.step(action)

        if done:
            timeEvolutionPearson.append(pearsonCorrelationOfBids(env.opp_utility_function.getDomain(),env.opp_utility_function.getUtility,agent.opponent_model.get_predicted_utility))
            print("Pearson Correlation of Bids:" + str(timeEvolutionPearson))
            rewards.append(reward)
            print("Reward:" + str(reward) + ", Opponent's reward:" + str(opp_reward) + ", reached aggrement after " + str(step) + " step(s)")
            plt.title("Accuracy of the Smith Frequency Model")
            plt.xlabel("Number of exchanged bids")
            plt.ylabel("Pearson correlation of bids")
            plt.plot(np.append(np.arange(0,step,sampleFrequency),step),timeEvolutionPearson)
            plt.draw()
            opp_rewards.append(opp_reward)
            break
        if step % sampleFrequency == 0:
            timeEvolutionPearson.append(pearsonCorrelationOfBids(env.opp_utility_function.getDomain(),env.opp_utility_function.getUtility,agent.opponent_model.get_predicted_utility))
        step += 1
    plt.show()

# print results
print(f"Average reward: {sum(rewards)/len(rewards)}")
print(f"Average opponent reward: {sum(opp_rewards)/len(opp_rewards)}")