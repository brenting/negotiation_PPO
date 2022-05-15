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
    SelfPlayAgent
)

# collect domains and opponents for trainig (don't initialise the opponents)

domains = get_domains("environment/domains/train")
opponents = (
    # BoulwareAgent,
    # ConcederAgent,
    # HardlinerAgent,
    # LinearAgent,
    # RandomAgent,
    # StupidAgent,
    SelfPlayAgent,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=1000)
agent = PPOAgent()

# set checkpoint path for intermediate model checkpoints
checkpoint_path = "checkpoint.pkl"

# train and save agent
agent.train(env=env, time_budget_sec=250000, checkpoint_path=checkpoint_path)
agent.save(checkpoint_path)