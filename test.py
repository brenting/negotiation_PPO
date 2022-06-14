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
    CSE3210,
)
    
# collect domains and opponents for testing (don't initialise the opponents)
domains = get_domains("environment/domains/test")

domains = domains

opponents = (
    HardlinerAgent,
    #BoulwareAgent,
    #ConcederAgent,
    #LinearAgent,
    # BoulwareAgent,
    #LinearAgent,
    # RandomAgent,
    # StupidAgent,
    # CSE3210.Agent2,
    # CSE3210.Agent3,
    # CSE3210.Agent7,
    # CSE3210.Agent11,
    # CSE3210.Agent14,
    #CSE3210.Agent18,
    # CSE3210.Agent19,
    # CSE3210.Agent22,
    # CSE3210.Agent24,
    # CSE3210.Agent25,
    #CSE3210.Agent26,
    # CSE3210.Agent27,
    # CSE3210.Agent29,
    # CSE3210.Agent32,
    # CSE3210.Agent33,
    #CSE3210.Agent41,
    #CSE3210.Agent43,
    # CSE3210.Agent50,
    #CSE3210.Agent52,
    # CSE3210.Agent55,
    # CSE3210.Agent58,
    # CSE3210.Agent61,
    #CSE3210.Agent64,
    #CSE3210.Agent67,
    #CSE3210.Agent68,
    #CSE3210.Agent70,
    #CSE3210.Agent78,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000, seed=42)
agent = PPOAgent.load("checkpoint.pkl")

# test on 50 random negotiation sessions and gather average results
rewards = []
opp_rewards = []
sumSmith = []
totalS=[]
sumPerceptron = []
totalP = []
result = []

N = 1

exp = OpponentModellingExperiment(opponents,N)

for round in range(N):
    obs = env.reset(agent)
    exp.reset(env)
    done = False
    opp_reward = (float)(env.opp_utility_function.getUtility(obs.getBid()))
    reward = (float)(env.my_utility_function.getUtility(obs.getBid()))
    step = 0
    estimed_opp_Smith = []
    estimed_opp_Perceptron = []
    while not done:
        step += 1
        #print("1",env.opp_utility_function.getUtility(obs.getBid()), agent.opponent_model2.get_predicted_utility(obs.getBid()), agent.opponent_model3.get_predicted_utility(obs.getBid()))
        action = agent.select_action(obs, training=False, estimatedUtiliy = opp_reward)
        exp.saveModels(agent, reward, opp_reward)
        #print("2",env.opp_utility_function.getUtility(obs.getBid()), agent.opponent_model2.get_predicted_utility(obs.getBid()), agent.opponent_model3.get_predicted_utility(obs.getBid()))
        obs, reward, done, opp_reward = env.step(action)
        if done:
            rewards.append(reward)
            print(f"Reward for round {round+1}/{N}: {reward}, Opponent's reward: {opp_reward} , reached aggrement after {step} step(s)")
            opp_rewards.append(opp_reward)
            exp.saveResults(env)
            break

    


# print results
print(f"Average reward: {sum(rewards)/len(rewards)}")
print(f"Average opponent reward: {sum(opp_rewards)/len(opp_rewards)}")
