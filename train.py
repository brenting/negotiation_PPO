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
)

domains = get_domains("environment/domains/train")

opponents = (
    ConcederAgent,
    HardlinerAgent,
    LinearAgent,
    RandomAgent,
    StupidAgent,
)

# Parallel environments
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)

agent = PPOAgent()

checkpoint_path = "checkpoint.pkl"

agent.learn(env=env, time_budget_sec=25000, checkpoint_path=checkpoint_path)
agent.save(checkpoint_path)