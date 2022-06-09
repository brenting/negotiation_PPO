from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from agent.ppo_agent import PPOAgent
from experiment.experiment import OpponentModellingExperiment

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

domains = domains[2:5]

opponents = (
    BoulwareAgent,
    HardlinerAgent,
    LinearAgent,
    ConcederAgent
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)
agent = PPOAgent.load("checkpoint.pkl")

# test on 50 random negotiation sessions and gather average results
rewards = []
opp_rewards = []
sumSmith = []
totalS=[]
sumPerceptron = []
totalP = []
result = []

N = 50

exp = OpponentModellingExperiment(opponents,N)

for round in range(N):
    obs = env.reset(agent)
    done = False
    opp_reward = (float)(env.opp_utility_function.getUtility(obs.getBid()))
    reward = (float)(env.my_utility_function.getUtility(obs.getBid()))
    step = 0
    estimed_opp_Smith = []
    estimed_opp_Perceptron = []
    while not done:
        print(reward,opp_reward)
        step += 1
        action = agent.select_action(obs,estimatedUtiliy = opp_reward)
        obs, reward, done, opp_reward = env.step(action)
        #making a deep copy of the estimated opponent model(s)
        exp.saveModels(agent)
        if done:
            rewards.append(reward)
            print(f"Reward for round {round+1}/{N}: {reward}, Opponent's reward: {opp_reward} , reached aggrement after {step} step(s)")
            opp_rewards.append(opp_reward)
            exp.saveResults(env)
            break

    


# print results
print(f"Average reward: {sum(rewards)/len(rewards)}")
print(f"Average opponent reward: {sum(opp_rewards)/len(opp_rewards)}")
