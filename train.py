from agent.utils.ppo import PPO
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

# collect domains and opponents for trainig (don't initialise the opponents)
domains = get_domains("environment/domains/train")
opponents = (
    ConcederAgent,
    HardlinerAgent,
    BoulwareAgent,
    LinearAgent,
    RandomAgent,
    StupidAgent,
    # CSE3210.Agent2,
    # CSE3210.Agent3,
    # CSE3210.Agent7,
    # CSE3210.Agent11,
    # CSE3210.Agent14,
    # CSE3210.Agent18,
    # CSE3210.Agent19,
    # CSE3210.Agent22,
    # CSE3210.Agent24,
    # CSE3210.Agent25,
    # CSE3210.Agent26,
    # CSE3210.Agent27,
    # CSE3210.Agent29,
    # CSE3210.Agent32,
    # CSE3210.Agent33,
    CSE3210.Agent41,
    CSE3210.Agent43,
    CSE3210.Agent50,
    CSE3210.Agent52,
    CSE3210.Agent55,
    CSE3210.Agent58,
    CSE3210.Agent61,
    CSE3210.Agent64,
    CSE3210.Agent67,
    # CSE3210.Agent68,
    # CSE3210.Agent70,
    CSE3210.Agent78,
)

PPO_PARAMETERS = {
    "state_dim": 15,  # dimension of state space
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
# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000, seed=42)
agent = PPOAgent(PPO_PARAMETERS=PPO_PARAMETERS, issue_values=True, private_info=True, exact_derived=True, inexact_derived=True)

# set checkpoint path for intermediate model checkpoints
checkpoint_path = "checkpoint_all.pkl"

# train and save agent
agent.train(env=env, time_budget_sec=14400, checkpoint_path=checkpoint_path)
agent.save(checkpoint_path)
