import time
from typing import cast

import numpy as np
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.simplerunner.NegoRunner import StdOutReporter
from tudelft_utilities_logging.Reporter import Reporter

from agent.utils.opponent_model import OpponentModel


class BaselineAgent():
    def __init__(self, report: Reporter = None):
        # super().__init__(reporter)
        self.domain: Domain = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.settings: Settings = None

        self.opponent_model: OpponentModel = None
        self.all_bids = None

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent for geniusweb agents.
        It is kept in the PPO agent for compatibility reasons.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), StdOutReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            self.all_bids = AllBidsList(self.domain)
            profile_connection.close()

            self.all_bids = sorted(self.all_bids, key=lambda x: self.get_utility(x),
                                   reverse=False)

    def select_action(self, obs: Offer, training=True) -> Action:
        """Method to return an action when it is our turn.

        Args:
            obs (Offer): Observation in the form of the offer made by the opponent.

        Returns:
            Action: Our action as a response on the Offer.
        """
        progress = self.progress.get(time.time() * 1000)
        self._acceptable_utility = (0.7 + (1 - progress) * 0.3)

        if obs is None or obs.getBid() is None:
            received_bid = None
            received_util = 0.0
        else:
            received_bid = obs.getBid()
            received_util = float(self.get_utility(received_bid))

        # return Accept if the reveived offer is better than our goal
        # print(util_goals)
        if self._acceptable_utility < received_util:
            return Accept(self.me, received_bid)

        # find a good bid based on the utility goals
        bid = self.findBid()

        assert bid, "bid cannot be None"

        action = Offer(self.me, bid)
        # if action.getBid() is None:
        #     print("Agent 1 offering: " + str(action))
        # print("Agent 1 offering: " + str(action))

        return action

    def findBid(self) -> Bid:
        min_idx = self.binary_search(self.all_bids, self._acceptable_utility)
        bid = np.random.choice(self.all_bids[min_idx:])
        # print(self.get_utility(bid), self.target, min_idx)
        return bid

    def binary_search(self, data, val):
        lo, hi = 0, len(data) - 1
        best_ind = lo
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if self.get_utility(data[mid]) < val:
                lo = mid + 1
            elif self.get_utility(data[mid]) > val:
                hi = mid - 1
            else:
                best_ind = mid
                break
            # check if data[mid] is closer to val than data[best_ind]
            if abs(self.get_utility(data[mid]) - val) < abs(self.get_utility(data[best_ind]) - val):
                best_ind = mid
        return best_ind

    def get_utility(self, bid: Bid) -> float:
        """returns utility value of bid for our agent"""
        return float(self.profile.getUtility(bid))
