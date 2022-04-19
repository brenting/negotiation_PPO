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

domains = get_domains("environment/domains/test")

opponents = (
    BoulwareAgent,
)

# Parallel environments
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)

agent = PPOAgent.load("checkpoint.pkl")

for _ in range(5):
    obs = env.reset(agent)
    done = False
    while not done:
        action = agent.select_action(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            print(reward)
            break
