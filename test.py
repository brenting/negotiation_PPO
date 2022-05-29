import copy
from math import sqrt
from numpy import average
from agent import ppo_agent
from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from agent.utils.pearson_correlation import PearsonCorrelation
from agent.ppo_agent import PPOAgent
import matplotlib.pyplot as plt
import numpy as np


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
    ConcederAgent,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)
agent = PPOAgent.load("checkpoint.pkl")

# test on 50 random negotiation sessions and gather average results
rewards = []
opp_rewards = []
for _ in range(50):
    obs = env.reset(agent)
    done = False
    step = 0
    estimed_opp_Smith = []
    estimed_opp_Perceptron = []
    sampleFrequency = 1

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, opp_reward = env.step(action)
        estimed_opp_Smith.append(copy.deepcopy(agent.opponent_model))
        estimed_opp_Perceptron.append(copy.deepcopy(agent.opponent_model2))
        if done:
            rewards.append(reward)
            print("Reward:" + str(reward) + ", Opponent's reward:" + str(opp_reward) + ", reached aggrement after " + str(step) + " step(s)")
            opp_rewards.append(opp_reward)
            break
        step += 1

    accuracySmith = []
    accuracyPerceptron = []
    pc = PearsonCorrelation(env.opp_utility_function.getDomain())

    for f in estimed_opp_Smith:
        accuracySmith.append(pc.pearsonCorrelationOfBids(env.opp_utility_function.getUtility,f.get_predicted_utility))
    for f in estimed_opp_Perceptron:    
        accuracyPerceptron.append(pc.pearsonCorrelationOfBids(env.opp_utility_function.getUtility,f.get_predicted_utility))
    
    plt.title("The evolution of the accuracy over the negotiation session")
    plt.xlabel("Number of exchanged bids")
    plt.ylabel("Pearson correlation of bids")
    plt.plot(np.append(np.arange(0,step,sampleFrequency),step),accuracySmith,label = "Smith Model")
    plt.plot(np.append(np.arange(0,step,sampleFrequency),step),accuracyPerceptron,label = "Perceptron Model")
    plt.legend()
    plt.draw()
    plt.show()


# print results
print(f"Average reward: {sum(rewards)/len(rewards)}")
print(f"Average opponent reward: {sum(opp_rewards)/len(opp_rewards)}")