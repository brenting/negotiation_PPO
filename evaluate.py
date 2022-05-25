import torch

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


def evaluate(agent):
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
    for _ in range(50):
        obs = env.reset(agent)
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, opp_reward = env.step(action)
            if done:
                # print("Done, reward: " + str(reward) + " " + str(opp_reward))
                rewards.append(reward)
                opp_rewards.append(opp_reward)
                break

    # print results
    print(f"Average reward: {sum(rewards)/len(rewards)}")
    print(f"Average opponent reward: {sum(opp_rewards)/len(opp_rewards)}")
