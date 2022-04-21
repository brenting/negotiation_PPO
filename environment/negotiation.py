import random
import time
from datetime import datetime
from typing import Optional

import gym
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
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

        self.opponent = None
        self.my_utility_function = None
        self.opp_utility_function = None

    def step(self, action: Inform) -> tuple[Action, float, bool, None]:
        if self.progress.get(time.time() * 1000) == 1:
            return None, 0.0, True, None  # observation, reward, done, info

        self.opponent.notifyChange(ActionDone(action))

        if isinstance(action, Accept):
            bid = action.getBid()
            agreements = Agreements({self.opponent_ID: bid, self.my_ID: bid})
            self.opponent.notifyChange(Finished(agreements))
            my_reward = float(self.my_utility_function.getUtility(action.getBid()))
            opp_reward = float(self.opp_utility_function.getUtility(action.getBid()))
            return None, my_reward, True, opp_reward  # observation, reward, done, info

        observation: Action = self.opponent.notifyChange(YourTurn())

        if self.progress.get(time.time() * 1000) == 1:
            return None, 0.0, True, None  # observation, reward, done, info

        if isinstance(observation, Accept):
            bid = observation.getBid()
            agreements = Agreements({self.opponent_ID: bid, self.my_ID: bid})
            self.opponent.notifyChange(Finished(agreements))
            my_reward = float(self.my_utility_function.getUtility(action.getBid()))
            opp_reward = float(self.opp_utility_function.getUtility(action.getBid()))
            return None, my_reward, True, opp_reward  # observation, reward, done, info

        return observation, 0.0, False, None  # observation, reward, done, info

    def reset(self, my_agent):
        self.opponent: DefaultParty = random.choice(self.opponents)(VoidReporter())
        domain = list(random.choice(self.domains))
        random.shuffle(domain)

        self.opp_utility_function = get_utility_function(domain[0])
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

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        ...

    def close(self):
        self.opponent = None
        self.my_utility_function = None
        self.opp_utility_function = None


class VoidReporter(Reporter):
	def log(self, level:int , msg:str, exc:Optional[BaseException]=None):
		pass
