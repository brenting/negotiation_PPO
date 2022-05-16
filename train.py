from environment.negotiation import NegotiationEnv
from environment.domains import get_domains
from agent.acceptance_agent import AcceptanceAgent
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
from utils.plot_trace import plot_training

domains = get_domains("environment/domains/train") #[0]
opponents = (
    ConcederAgent,
    HardlinerAgent,
    LinearAgent,
    RandomAgent,
    # StupidAgent,
)

# create environment and PPO agent
env = NegotiationEnv(domains=domains, opponents=opponents, deadline_ms=10000)
agent = AcceptanceAgent()

# set checkpoint path for intermediate model checkpoints
checkpoint_path = "checkpoint.pkl"

# train and save agent
agent.train(env=env, time_budget_sec=3600, checkpoint_path=checkpoint_path)
agent.save(checkpoint_path)
plot_training(agent.rewards, "results/training_plot.html")
