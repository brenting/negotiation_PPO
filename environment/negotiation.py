import random
import time
from datetime import datetime
from typing import Optional

import gym
from geniusweb.actions.Offer import Offer
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.EndNegotiation import EndNegotiation
from geniusweb.actions.PartyId import PartyId
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Agreements import Agreements
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from geniusweb.references.ProfileRef import ProfileRef
from geniusweb.references.ProtocolRef import ProtocolRef
from tudelft_utilities_logging.Reporter import Reporter
from uri.uri import URI

from environment.domains import get_utility_function
from utils.plot_trace import plot_nash_kalai_pareto


class NegotiationEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            domains: tuple[tuple[URI, URI]],
            opponents: tuple[DefaultParty],
            deadline_ms: int,
    ):
        super().__init__()

        self.domains = domains
        self.opponents = opponents
        self.deadline_ms = deadline_ms
        self.current_domain = None

        self.opponent = None
        self.my_utility_function = None
        self.opp_utility_function = None
        self.trace = {"actions": []}
        self.my_domain = None
        self.opp_domain = None

    def step(self, action: Inform) -> tuple[Action, float, bool, float]:
        if self.progress.get(time.time() * 1000) == 1:
            self.opponent.notifyChange(EndNegotiation)
            return None, 0.0, True, 0.0  # observation, reward, done, info

        self.opponent.notifyChange(ActionDone(action))

        if isinstance(action, Accept):
            self.append_trace(action, None)

            bid = action.getBid()
            agreements = Agreements({self.opponent_ID: bid, self.my_ID: bid})
            self.opponent.notifyChange(Finished(agreements))
            my_reward = float(self.my_utility_function.getUtility(action.getBid()))
            opp_reward = float(self.opp_utility_function.getUtility(action.getBid()))
            return None, my_reward, True, opp_reward  # observation, reward, done, info

        observation: Action = self.opponent.notifyChange(YourTurn())

        if self.progress.get(time.time() * 1000) == 1:
            self.opponent.notifyChange(Finished)
            return None, 0.0, True, 0.0  # observation, reward, done, info

        self.append_trace(action, observation)

        if isinstance(observation, Accept):

            bid = observation.getBid()
            agreements = Agreements({self.opponent_ID: bid, self.my_ID: bid})
            self.opponent.notifyChange(Finished(agreements))
            my_reward = float(self.my_utility_function.getUtility(action.getBid()))
            opp_reward = float(self.opp_utility_function.getUtility(action.getBid()))
            return None, my_reward, True, opp_reward  # observation, reward, done, info

        return observation, 0.0, False, 0.0  # observation, reward, done, info

    def reset(self, my_agent):
        self.opponent: DefaultParty = random.choice(self.opponents)(VoidReporter())
        domain = list(random.choice(self.domains))
        random.shuffle(domain)
        self.trace = {"actions": []}
        self.current_domain = domain
        self.opp_utility_function = get_utility_function(domain[0])
        self.my_domain = domain[1]
        self.opp_domain = domain[0]
        self.my_utility_function = get_utility_function(domain[1])

        self.progress = ProgressTime(self.deadline_ms, datetime.now())

        protocol = ProtocolRef(URI("SAOP"))

        self.opponent_ID = PartyId(type(self.opponent).__name__)
        self.my_ID = PartyId(type(my_agent).__name__)

        opp_settings = Settings(
            self.opponent_ID,
            ProfileRef(domain[0]),
            protocol,
            self.progress,
            Parameters(),
        )
        my_settings = Settings(
            self.my_ID,
            ProfileRef(domain[1]),
            protocol,
            self.progress,
            Parameters(),
        )

        self.opponent.notifyChange(opp_settings)
        my_agent.notifyChange(my_settings)

        observation = self.opponent.notifyChange(YourTurn())

        # write e value to file
        # f = open("evalue.txt", "w")
        # try:
        #     e = self.opponent.getE()
        # except AttributeError:
        #     e = 0.5
        # f.write(str(e))

        return observation  # reward, done, info can't be included

    def append_trace(self, my_action, opp_action):

        if isinstance(my_action, Accept) and opp_action is None:
            my_reward = float(self.my_utility_function.getUtility(my_action.getBid()))
            opp_reward = float(self.opp_utility_function.getUtility(my_action.getBid()))
            self.trace["actions"].append(
                {"Accept": {"actor": "me", "utilities": {"me": my_reward, "opponent": opp_reward}}})

        if isinstance(my_action, Offer):
            my_reward = float(self.my_utility_function.getUtility(my_action.getBid()))
            opp_reward = float(self.opp_utility_function.getUtility(my_action.getBid()))
            self.trace["actions"].append(
                {"Offer": {"actor": "me", "utilities": {"me": my_reward, "opponent": opp_reward}}})

        if isinstance(opp_action, Offer):
            my_reward = float(self.my_utility_function.getUtility(opp_action.getBid()))
            opp_reward = float(self.opp_utility_function.getUtility(opp_action.getBid()))
            self.trace["actions"].append(
                {"Offer": {"actor": "opponent", "utilities": {"me": my_reward, "opponent": opp_reward}}})

        if isinstance(opp_action, Accept):
            my_reward = float(self.my_utility_function.getUtility(opp_action.getBid()))
            opp_reward = float(self.opp_utility_function.getUtility(opp_action.getBid()))
            self.trace["actions"].append(
                {"Accept": {"actor": "opponent", "utilities": {"me": my_reward, "opponent": opp_reward}}})

    def render(self, mode="human"):
        ...

    def close(self):
        self.opponent = None
        self.my_utility_function = None
        self.opp_utility_function = None


class VoidReporter(Reporter):
    def log(self, level: int, msg: str, exc: Optional[BaseException] = None):
        pass
