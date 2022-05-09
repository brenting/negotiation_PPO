from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from agent.ppo_agent import PPOAgent
from agent.acceptance_agent import AcceptanceAgent
from environment.opponents import (
    BoulwareAgent,
    ConcederAgent,
    HardlinerAgent,
    LinearAgent,
    RandomAgent,
    StupidAgent,
)

# collect domains and opponents for trainig (don't initialise the opponents)
from utils.plot_trace import plot_training

domains = get_domains("environment/domains/train") #[0]
opponents = (
    ConcederAgent,
    HardlinerAgent,
    LinearAgent,
    # RandomAgent,
    # StupidAgent,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)
agent = AcceptanceAgent()

# set checkpoint path for intermediate model checkpoints
checkpoint_path = "checkpoint.pkl"

# train and save agent
rewards = agent.train(env=env, time_budget_sec=60, checkpoint_path=checkpoint_path)
print(rewards)
agent.save(checkpoint_path)
plot_training(rewards, "results/training_plot.html")
