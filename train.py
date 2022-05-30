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
    # ConcederAgent,
    # HardlinerAgent,
    # BoulwareAgent,
    # LinearAgent,
    # RandomAgent,
    # StupidAgent,
    # CSE3210.Agent2,
    CSE3210.Agent3,
    CSE3210.Agent7,
    CSE3210.Agent11,
    CSE3210.Agent14,
    CSE3210.Agent18,
    CSE3210.Agent19,
    CSE3210.Agent22,
    CSE3210.Agent24,
    CSE3210.Agent25,
    # CSE3210.Agent26,
    CSE3210.Agent27,
    CSE3210.Agent29,
    CSE3210.Agent32,
    CSE3210.Agent33,
    # CSE3210.Agent41,
    # CSE3210.Agent43,
    CSE3210.Agent50,
    # CSE3210.Agent52,
    CSE3210.Agent55,
    CSE3210.Agent58,
    CSE3210.Agent61,
    # CSE3210.Agent64,
    # CSE3210.Agent67,
    # CSE3210.Agent68,
    # CSE3210.Agent70,
    # CSE3210.Agent78,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)
agent = PPOAgent()

# set checkpoint path for intermediate model checkpoints
checkpoint_path = "checkpoint.pkl"

# train and save agent
agent.train(env=env, time_budget_sec=25000, checkpoint_path=checkpoint_path)
agent.save(checkpoint_path)