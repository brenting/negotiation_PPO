import matplotlib.pyplot as plt

x = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
reward = [0.501, 0.485, 0.530, 0.547, 0.492, 0.568, 0.579, 0.548]
opp_reward = [0.723, 0.727, 0.819, 0.844, 0.833, 0.844, 0.856, 0.837]

plt.plot(x, reward, c="blue", marker="o", label="Reward")
plt.plot(x, opp_reward, c="red", marker="o", label="Opponent Reward")

plt.title("Base agent")
plt.legend(loc="lower left")
plt.xlabel("Time in hours")
plt.ylabel("Reward")
plt.ylim(0, 1)
plt.show()
plt.close()

private_reward = [0.550, 0.561, 0.558, 0.547, 0.540, 0.601, 0.585, 0.570]
private_minus_extra_feature_reward = [0.632, 0.621, 0.617, 0.576, 0.584, 0.553, 0.535, 0.553]
opp_private_minus_extra_feature_reward = [0.813, 0.761, 0.781, 0.799, 0.866, 0.863, 0.837, 0.861]
opp_reward = [0.752, 0.733, 0.742, 0.743, 0.817, 0.765, 0.794, 0.743]

plt.plot(x, private_reward, c="blue", marker="o", label="Reward")
plt.plot(x, private_minus_extra_feature_reward, c="purple", marker="o", label="ignored features Reward")
plt.plot(x, reward, c="red", marker="o", label="Base agent Reward")

plt.title("Private Info agent analysis")
plt.legend(loc="lower left")
plt.xlabel("Time in hours")
plt.ylabel("Reward")
plt.ylim(0, 1)
plt.show()
plt.close()


issue_reward = [0.356, 0.306, 0.327, 0.579, 0.597, 0.640, 0.668, 0.664]
issue_minus_extra_features_reward = [0.472, 0.460, 0.456, 0.488, 0.483, 0.504, 0.503, 0.471]
opp_reward = [0.700, 0.689, 0.673, 0.573, 0.525, 0.553, 0.591, 0.561]

plt.plot(x, issue_reward, c="blue", marker="o", label="Reward")
plt.plot(x, issue_minus_extra_features_reward, c="purple", marker="o", label="ignored features Reward")
plt.plot(x, reward, c="red", marker="o", label="Base agent Reward")

plt.title("Issue Values agent analysis")
plt.legend(loc="lower left")
plt.xlabel("Time in hours")
plt.ylabel("Reward")
plt.ylim(0, 1)
plt.show()
plt.close()

exact_derived_reward = [0.617, 0.659, 0.615, 0.580, 0.544, 0.618, 0.603, 0.603]
exact_derived_minus_extra_features_reward = [0.543, 0.584, 0.602, 0.546, 0.541, 0.527, 0.499, 0.520]
opp_reward = [0.367, 0.374, 0.338, 0.283, 0.165, 0.323, 0.343, 0.335]

plt.plot(x, exact_derived_reward, c="blue", marker="o", label="Reward")
plt.plot(x, exact_derived_minus_extra_features_reward, c="purple", marker="o", label="ignored features Reward")
plt.plot(x, reward, c="red", marker="o", label="Base agent Reward")

plt.title("Exact Derived agent analysis")
plt.legend(loc="lower left")
plt.xlabel("Time in hours")
plt.ylabel("Reward")
plt.ylim(0, 1)
plt.show()
plt.close()

inexact_derived_reward = [0.330, 0.499, 0.578, 0.593, 0.627, 0.582, 0.617, 0.520]
inexact_derived_minus_extra_features_reward = [0.544, 0.519, 0.561, 0.612, 0.591, 0.577, 0.500, 0.527]
opp_reward = [0.092, 0.207, 0.277, 0.344, 0.391, 0.310, 0.334, 0.277]

plt.plot(x, inexact_derived_reward, c="blue", marker="o", label="Reward")
plt.plot(x, inexact_derived_minus_extra_features_reward, c="purple", marker="o", label="ignored features Reward")
plt.plot(x, reward, c="red", marker="o", label="Base agent Reward")

plt.title("Inexact Derived agent analysis")
plt.legend(loc="lower left")
plt.xlabel("Time in hours")
plt.ylabel("Reward")
plt.ylim(0, 1)
plt.show()
plt.close()


all_agents = [0.612, 0.641, 0.638, 0.648, 0.664, 0.653, 0.620, 0.633]
opp_reward = [0.343, 0.483, 0.481, 0.513, 0.533, 0.505, 0.472, 0.486]

plt.plot(x, all_agents, c="blue", marker="o", label="Reward")
plt.plot(x, opp_reward, c="red", marker="o", label="Opponent Reward")

plt.title("Combined agent")
plt.legend(loc="lower left")
plt.xlabel("Time in hours")
plt.ylabel("Reward")
plt.ylim(0, 1)
plt.show()
plt.close()

plt.plot(x, reward, c="blue", marker="o", label="Base reward")
plt.plot(x, private_reward, c="red", marker="o", label="Private Agent Reward")
plt.plot(x, issue_reward, c="purple", marker="o", label="Issue Agent Reward")
plt.plot(x, exact_derived_reward, c="green", marker="o", label="Exact Agent Reward")
plt.plot(x, inexact_derived_reward, c="black", marker="o", label="Inexact Agent Reward")

plt.title("All agents")
plt.legend(loc="lower left")
plt.xlabel("Time in hours")
plt.ylabel("Reward")
plt.ylim(0, 1)
plt.show()
plt.close()

plt.plot(x, all_agents, c="red", marker="o", label="Combined Agent Reward")
plt.plot(x, reward, c="blue", marker="o", label="Base reward")

plt.title("Base agent compared to Combined agent")
plt.legend(loc="lower left")
plt.xlabel("Time in hours")
plt.ylabel("Reward")
plt.ylim(0, 1)
plt.show()
plt.close()