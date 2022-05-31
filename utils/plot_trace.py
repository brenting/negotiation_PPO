from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import os
import plotly.express as px


def plot_nash_kalai_pareto(results_trace: dict, nash_point, kalai_point, pareto_utilities, plot_file: str, switch):
    # Negotiation trace
    utilities = defaultdict(lambda: defaultdict(lambda: {"x": [], "y": [], "bids": []}))
    accept = {"x": [], "y": [], "bids": []}

    if switch:
        agentsNames = ["opponent", "me"]
    else:
        agentsNames = ["me", "opponent"]

    # print(results_trace["actions"][len(results_trace["actions"]) - 1])
    for index, action in enumerate(results_trace["actions"], 1):
        if "Offer" in action:
            offer = action["Offer"]
            actor = offer["actor"]

            if actor not in agentsNames:
                agentsNames.append(actor)
                print(actor)

            for agent, util in offer["utilities"].items():
                utilities[agent][actor]["x"].append(index)
                utilities[agent][actor]["y"].append(util)
                # utilities[agent][actor]["bids"].append(offer["bid"]["issuevalues"])

        elif "Accept" in action:
            offer = action["Accept"]
            index -= 1
            for agent, util in offer["utilities"].items():
                if agent == agentsNames[0]:
                    accept["x"].append(util)
                elif agent == agentsNames[1]:
                    accept["y"].append(util)
                # accept["bids"].append(offer["bid"]["issuevalues"])

    agent1 = agentsNames[0]
    agent2 = agentsNames[1]

    xAxes = utilities[agent1][agent1]["y"]
    yAxes = utilities[agent2][agent1]["y"]

    xAxes2 = utilities[agent1][agent1]["y"]
    yAxes2 = utilities[agent2][agent1]["y"]

    text = []
    for i in range(0, len(xAxes)):
        text.append(i)

    # Pareto frontier part
    xAxes_pareto = []
    yAxes_pareto = []

    for tuple in pareto_utilities:
        xAxes_pareto.append(tuple[0])
        yAxes_pareto.append(tuple[1])

    # Add everything to figure
    fig = go.Figure()

    # Nash point
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=[nash_point[0]],
            y=[nash_point[1]],
            name="nash",
            marker={"color": "yellow", "size": 15},
            hoverinfo="skip",
        )
    )

    # Kalai point
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=[kalai_point[0]],
            y=[kalai_point[1]],
            name="kalai",
            marker={"color": "red", "size": 15},
            hoverinfo="skip",
        )
    )
    # Agreement point
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=accept["x"],
            y=accept["y"],
            name="agreement",
            marker={"color": "purple", "size": 15},
            hoverinfo="skip",
        )
    )
    # Pareto frontier
    fig.add_trace(
        go.Scatter(
            mode="lines+markers",
            x=xAxes_pareto,
            y=yAxes_pareto,
            name="pareto_frontier",
            marker={"color": "blue"},
            hoverinfo="skip",
        )
    )

    # Negotiation trace
    fig.add_trace(
        go.Scatter(
            mode="lines+markers",
            x=xAxes,
            y=yAxes,
            name="negotiation_trace",
            marker={"color": "red"},
            hovertext=text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        # width=1000,
        height=800,
        legend={
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "left",
            "x": 0,
        },
    )

    fig.update_xaxes(title_text=agent1, range=[0, 1.01], ticks="outside")
    fig.update_yaxes(title_text=agent2, range=[0, 1.01], ticks="outside")
    fig.write_html(f"{os.path.splitext(plot_file)[0]}.html")


def plot_trace(results_trace: dict, plot_file: str):
    utilities = defaultdict(lambda: defaultdict(lambda: {"x": [], "y": [], "bids": []}))
    accept = {"x": [], "y": [], "bids": []}

    for index, action in enumerate(results_trace["actions"], 1):
        if "Offer" in action:
            offer = action["Offer"]
            actor = offer["actor"]

            for agent, util in offer["utilities"].items():
                utilities[agent][actor]["x"].append(index)
                utilities[agent][actor]["y"].append(util)
            # utilities[agent][actor]["bids"].append(offer["bid"]["issuevalues"])

        elif "Accept" in action:
            offer = action["Accept"]
            index -= 1
            for agent, util in offer["utilities"].items():
                accept["x"].append(index)
                accept["y"].append(util)
                # accept["bids"].append(offer["bid"]["issuevalues"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=accept["x"],
            y=accept["y"],
            name="agreement",
            marker={"color": "green", "size": 30},
            hoverinfo="skip",
        )
    )

    color = {0: "red", 1: "blue"}
    for i, (agent, data) in enumerate(utilities.items()):
        for actor, utility in data.items():
            name = "_".join(agent.split("_")[-2:])
            text = []
            for bid, util in zip(utility["bids"], utility["y"]):
                text.append(
                    "<br>".join(
                        [f"<b>utility: {util:.3f}</b><br>"]
                        # + [f"{i}: {v}" for i, v in bid.items()]
                    )
                )
            fig.add_trace(
                go.Scatter(
                    mode="lines+markers" if agent == actor else "markers",
                    x=utilities[agent][actor]["x"],
                    y=utilities[agent][actor]["y"],
                    name=f"{name} offered" if agent == actor else f"{name} received",
                    legendgroup=agent,
                    marker={"color": color[i]},
                    hovertext=text,
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        # width=1000,
        height=800,
        legend={
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "left",
            "x": 0,
        },
    )
    fig.update_xaxes(title_text="round", range=[0, index + 1], ticks="outside")
    fig.update_yaxes(title_text="utility", range=[0, 1], ticks="outside")
    fig.write_html(f"{os.path.splitext(plot_file)[0]}.html")

def plot_training(rewards: list, plot_file: str):
    fig = go.Figure()
    fig = px.scatter(x=np.arange(len(rewards)), y=rewards, trendline="ols")
    fig.show()
    fig.write_html(f"{os.path.splitext(plot_file)[0]}.html")

def distance_to_nash(results_trace: dict, nash_point, switch):
    reward = results_trace["actions"][len(results_trace["actions"]) - 1]['Accept']['utilities']['me']
    opp_reward = results_trace["actions"][len(results_trace["actions"]) - 1]['Accept']['utilities']['opponent']
    if switch:
        utils = np.array([opp_reward, reward])
    else:
        utils = np.array([reward, opp_reward])
    nash = np.array(nash_point)
    dist = np.linalg.norm(utils - nash)
    return dist
