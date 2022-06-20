from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from agent.ppo_agent import PPOAgent
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
opponents = (
    ConcederAgent,
    HardlinerAgent,
    BoulwareAgent,
    LinearAgent,
    RandomAgent,
    StupidAgent,
    CSE3210.Agent2,
    CSE3210.Agent3,
    CSE3210.Agent7,
    CSE3210.Agent11,
    CSE3210.Agent14,
    CSE3210.Agent18,
    CSE3210.Agent19,
    CSE3210.Agent22,
    CSE3210.Agent24,
    CSE3210.Agent25,
    CSE3210.Agent26,
    CSE3210.Agent27,
    CSE3210.Agent29,
    CSE3210.Agent32,
    CSE3210.Agent33,
    CSE3210.Agent41,
    # CSE3210.Agent43,
    # CSE3210.Agent50,
    # CSE3210.Agent52,
    # CSE3210.Agent55,
    # CSE3210.Agent58,
    # CSE3210.Agent61,
    # CSE3210.Agent64,
    # CSE3210.Agent67,
    # CSE3210.Agent68,
    # CSE3210.Agent70,
    # CSE3210.Agent78,
)

PPO_PARAMETERS = {
    "state_dim": 6,  # dimension of state space
    "action_dim": 2,  # dimension of action space
    "lr_actor": 0.0003,  # learning rate for actor network
    "lr_critic": 0.001,  # learning rate for critic network
    "gamma": 1,  # discount factor
    "K_epochs": 10,  # update policy for K epochs in one PPO update
    "eps_clip": 0.2,  # clip parameter for PPO
    "action_std": 0.6,  # starting std for action distribution (Multivariate Normal)
    "action_std_decay_rate": 0.05,  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    "min_action_std": 0.1,  # minimum action_std (stop decay after action_std <= min_action_std)
}

for j in range(8):
    # create environment and PPO agent
    env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000, seed=42)
    # issue_values and exact_derived got switched up
    agent = PPOAgent.load(str(j) + "checkpoint_inexact_derived.pkl", PPO_PARAMETERS)
    agent.set(issue_values=False, private_info=False, exact_derived=False, inexact_derived=True, test=True, PPO_PARAMETERS=PPO_PARAMETERS)

    # test on 50 random negotiation sessions and gather average results
    rewards = []
    opp_rewards = []
    for i in range(50):
        obs = env.reset(agent)
        done = False
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, done, opp_reward = env.step(action)
            if done:
                rewards.append(reward)
                opp_rewards.append(opp_reward)
                break

    # print results
    print(j)
    print(f"Average reward: {sum(rewards)/len(rewards)}")
    print(f"Average opponent reward: {sum(opp_rewards)/len(opp_rewards)}")
