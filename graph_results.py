from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import os
import plotly.express as px


def graph_results(xs, rewards, opp_rewards, rewards2, opp_rewards2):
    fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         mode="lines+markers",
    #         x=xs,
    #         y=rewards,
    #         name="Own utility",
    #         legendgroup="Own",
    #         marker={"color": "red"},
    #         # hovertext=text,
    #         hoverinfo="text",
    #     )
    # )
    #
    # fig.add_trace(
    #     go.Scatter(
    #         mode="lines+markers",
    #         x=xs,
    #         y=opp_rewards,
    #         name="Opponent utility",
    #         legendgroup="Own",
    #         marker={"color": "blue"},
    #         # hovertext=text,
    #         hoverinfo="text",
    #     )
    # )

    fig.add_trace(
        go.Scatter(
            mode="lines+markers",
            x=xs,
            y=rewards2,
            name="Own utility",
            legendgroup="Own",
            marker={"color": "red"},
            # hovertext=text,
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter(
            mode="lines+markers",
            x=xs,
            y=opp_rewards2,
            name="Opponent utility",
            legendgroup="Own",
            marker={"color": "blue"},
            # hovertext=text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        width=1000,
        height=500,
        # legend={
        #     "yanchor": "bottom",
        #     "y": 0.6,
        #     "xanchor": "left",
        #     "x": 0.01,
        # },
        title="Figure 1. Utility of the baseline model after different number of training iterations",
        # title="Figure 2. Utility of the self-play model after different number of training iterations",
        title_x=0.5,
        title_y=0.85,
    )
    fig.update_xaxes(title_text="iterations", range=[0, xs[-1] + 100], ticks="outside")
    fig.update_yaxes(title_text="utility", range=[0, 1], ticks="outside")
    fig.write_html(f"results/graph_results.html")


# graph_results([
#     100,
#     200,
#     300,
#     400,
#     500,
#     600,
#     700,
#     800,
#     900,
#     1000,
#     1100,
#     1200,
#     1300,
#     1400,
#     1500,
#     1600,
#     1700,
#     1800,
#     1900,
#     2000,
#     2100,
#     2200
# ], [
#     0.377060365046,
#     0.439120650768,
#     0.460394277688,
#     0.46640717384200014,
#     0.49125956212199995,
#     0.47428076961999993,
#     0.549124432638,
#     0.590922795508,
#     0.5961328583000001,
#     0.6445621460260003,
#     0.703095580688,
#     0.6935249882479999,
#     0.7353394314820001,
#     0.7353723075739999,
#     0.698899423088,
#     0.7213545450479998,
#     0.717602921206,
#     0.7677200196079998,
#     0.7446370292840001,
#     0.744909705796,
#     0.6813296771779999,
#     0.71389430353
# ], [
#     0.9346120635699999,
#     0.94812011526,
#     0.96126425169,
#     0.8814028569280002,
#     0.9008544411980001,
#     0.9212338225300002,
#     0.827637934962,
#     0.8153364595220001,
#     0.7757659107679999,
#     0.774145190134,
#     0.8141133308160002,
#     0.7516160260920001,
#     0.773004158934,
#     0.8155579821539998,
#     0.6806014012640001,
#     0.7001315876720002,
#     0.745466099886,
#     0.818694565242,
#     0.7783552999619998,
#     0.7901985641959999,
#     0.7663571600639999,
#     0.8046618073559999
# ])


graph_results([
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
    2100,
    2200
], [
    0.34309903051199997,
    0.3987268221559999,
    0.32915003116200003,
    0.3583840986680001,
    0.34823703945999995,
    0.363692788002,
    0.3885047962559999,
    0.46091417980599997,
    0.47590705471,
    0.56515056984,
    0.5714405218639999,
    0.614807278886,
    0.5976018945959998,
    0.7075983990779997,
    0.719216374376,
    0.7504976488940003,
    0.640014078408,
    0.740437303712,
    0.7539314333860001,
    0.7030740302580003,
    0.6824121113240001,
    0.7257637572100002,
], [
    0.972754198092,
    0.95414443516,
    0.951426337876,
    0.9531418124940001,
    0.9371187813420002,
    0.8159163545739997,
    0.8274311016720002,
    0.8625892403679998,
    0.7685597000280002,
    0.858386697504,
    0.8155799937820001,
    0.7520940572060002,
    0.7162446827080001,
    0.7761083922540003,
    0.748322120532,
    0.624892267586,
    0.550142721136,
    0.6581851775359999,
    0.612764136384,
    0.6018594060820002,
    0.5625422585600001,
    0.5796000792399999
], [
    0.402826586174,
    0.40159597108599987,
    0.43018881309799994,
    0.49563339062600015,
    0.5276386330859998,
    0.5111161569720001,
    0.46984145508,
    0.500048195518,
    0.5483440347939998,
    0.6057323571419999,
    0.6029303639220001,
    0.5792829384599999,
    0.6237613969479999,
    0.659144461254,
    0.6839080526860001,
    0.693806406202,
    0.7181231735880002,
    0.7103005692680001,
    0.735110959222,
    0.7022024276100002,
    0.6695456098679997,
    0.6907702368399999
], [
    0.9721422281,
    0.9191849081339999,
    0.90220645494,
    0.9114416049100003,
    0.9462998187300002,
    0.849297188022,
    0.809917309452,
    0.7646255063900002,
    0.81474298313,
    0.7707415903179999,
    0.75529587157,
    0.7423146021840001,
    0.7100621207359998,
    0.77029599166,
    0.7153925644019999,
    0.7359377687120001,
    0.7521225534820001,
    0.7510362965640002,
    0.7180511510260001,
    0.690151645562,
    0.67552747086,
    0.6330314532879999
])