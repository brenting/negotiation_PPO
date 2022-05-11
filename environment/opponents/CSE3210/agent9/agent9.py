import logging
import time
from decimal import Decimal
from random import randint
from typing import cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.opponentmodel import FrequencyOpponentModel
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.ValueSet import ValueSet
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressRounds import ProgressRounds
from tudelft_utilities_logging.Reporter import Reporter


class Agent9(DefaultParty):
    """
    Template agent that offers random bids until a bid with sufficient utility is offered.
    """

    def __init__(self, reporter: Reporter = None):
        super().__init__(reporter)
        self.getReporter().log(logging.INFO, "party is initialized")
        self._profile = None
        self._last_received_bid: Bid = None
        self._second_to_last_received_bid: Bid = None
        self._last_sent_bid: Bid = None
        self._bids_and_utils = dict()
        self._opponent_model = FrequencyOpponentModel.FrequencyOpponentModel.create()
        self._classification = 'concession'
        self._agent_util_change = 0
        self._opponent_util_change = 0
        self._opponent_id: PartyId = None
        self._bid_utils = []

    def notifyChange(self, info: Inform):
        """This is the entry point of all interaction with your agent after is has been initialised.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(info, Settings):
            self._settings: Settings = cast(Settings, info)
            self._me = self._settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self._progress: ProgressRounds = self._settings.getProgress()

            # the profile contains the preferences of the agent over the domain
            self._profile = ProfileConnectionFactory.create(
                info.getProfile().getURI(), self.getReporter()
            )

            # create a dict with all  bids and their utilities so we can
            # pick the bods woth the highest utils at the start
            domain = self._profile.getProfile().getDomain()
            all_bids = AllBidsList(domain)
            for b in all_bids:
                u = self._profile.getProfile().getUtility(b)
                self._bids_and_utils[u] = b

            # initialize the opponent model
            self._opponent_model = FrequencyOpponentModel.FrequencyOpponentModel.create()

        # ActionDone is an action send by an opponent (an offer or an accept)
        elif isinstance(info, ActionDone):
            action: Action = cast(ActionDone, info).getAction()
            self._opponent_id = action.getActor()
            # if it is an offer, set the last received bid
            if isinstance(action, Offer):
                self._last_received_bid = cast(Offer, action).getBid()

        # YourTurn notifies you that it is your turn to act
        elif isinstance(info, YourTurn):
            action = self._myTurn()
            if isinstance(self._progress, ProgressRounds):
                self._progress = self._progress.advance()
            return action

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(info, Finished):
            # terminate the agent MUST BE CALLED
            self.terminate()
        else:
            self.getReporter().log(
                logging.WARNING, "Ignoring unknown info " + str(info)
            )

    # lets the geniusweb system know what settings this agent can handle
    # leave it as it is for this course
    def getCapabilities(self) -> Capabilities:
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    # terminates the agent and its connections
    # leave it as it is for this course
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        super().terminate()
        if self._profile is not None:
            self._profile.close()
            self._profile = None

    #######################################################################################
    ########## THE METHODS BELOW THIS COMMENT ARE OF MAIN INTEREST TO THE COURSE ##########
    #######################################################################################

    # give a description of your agent
    def getDescription(self) -> str:
        return "Template agent for Collaborative AI course"

    # execute a turn
    def _myTurn(self):
        # check if the last received offer if the opponent is good enough
        if self._isGood(self._last_received_bid):
            # if so, accept the offer
            action = Accept(self._me, self._last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self._findBid()
            self._last_sent_bid = bid
            if bid is None:
                action = Accept(self._me, self._last_received_bid)
            else:
                action = Offer(self._me, bid)

        # send the action
        return action

    # method that checks if we would agree with an offer
    def _isGood(self, bid: Bid) -> bool:
        if bid is None:
            return False
        profile = self._profile.getProfile()
        domain = self._profile.getProfile().getDomain()

        self._bid_utils.append(profile.getUtility(bid))

        if domain is None:
            print("domain is none")
        if self._profile.getProfile() is None:
            print("getProfile() is none")
        if self._profile is None:
            print("_profile is none")

        progress = self._progress.get(time.time() * 1000)
        # check if utility of received bid is higher than last sent bid and accept if true

        # update opponent model
        if self._progress.get((int)(time.time() * 1000)) <= 0.01:
            self._opponent_model = self._opponent_model.With(domain, bid)
        else:
            action = Offer(self._opponent_id, bid)
            self._opponent_model = self._opponent_model.WithAction(action, self._progress)


        progress = self._progress.get(time.time())
        totalrounds = self._progress.getTotalRounds()
        remaining_rounds = totalrounds - progress * totalrounds
        current_round = progress * totalrounds
        max_bid = 0

        if progress >= 0.5:
            for i in range(int(remaining_rounds)):
                b = self._bid_utils[int(current_round - remaining_rounds + i - 1)]
                if b > max_bid:
                    max_bid = b

        if profile.getUtility(bid) > profile.getUtility(self._findBid()) or \
                profile.getUtility(bid) >= max_bid and progress > 0.85:
            return True



        # get the change in utility for our agent and the opponent between the tow last bids of the opponent
        # for classification purposes
        if self._second_to_last_received_bid is not None:
            self._agent_util_change = profile.getUtility(bid) - profile.getUtility(self._second_to_last_received_bid) \

            self._opponent_util_change = self._opponent_model.getUtility(bid) - \
                self._opponent_model.getUtility(self._second_to_last_received_bid)
        else:
            self._agent_util_change = 0
            self._opponent_util_change = 0

        # classify the opponents move to silent
        if self._agent_util_change == 0:
            if self._opponent_util_change == 0:
                self._classification = 'silent'

        # classify the opponents move to concession
        if self._agent_util_change > 0:
            if self._opponent_util_change < 0:
                self._classification = 'concession'

        # classify the opponents move to unfortunate
        if self._agent_util_change < 0:
            if self._opponent_util_change < 0:
                self._classification = 'unfortunate'

        # classify the opponents move to selfish
        if self._agent_util_change < 0:
            if self._opponent_util_change > 0:
                self._classification = 'selfish'

        # classify the opponents move to fortunate
        if self._agent_util_change > 0:
            if self._opponent_util_change > 0:
                self._classification = 'fortunate'

        # set second to last bid
        self._second_to_last_received_bid = bid

        return False

    def _findBid(self) -> Bid:
        # find bid with max util value
        bid = None
        profile = self._profile.getProfile()

        # first 2 percent bids are just highest util bids
        if self._progress.get(time.time()) < 0.02:
            max_util = max(self._bids_and_utils)
            bid = self._bids_and_utils.get(max_util)
            return bid

        # here we set the amout the util change matters,
        # if it is too high the offers will vary to much and if its too low they vary too little
        i = 0.0
        if self._classification in ['fortunate', 'unfortunate']:
            i = 0.25

        if self._classification in ['selfish', 'concession']:
            i = - 0.25

        # if the move of the opponent is silent (not changing his offer) do a random bid if the progress is beyond 50%
        # else do a bid that is equal to the last bid sent
        if self._classification in ['silent']:
            if self._progress.get(time.time()) >= 0.5:
                allBids = AllBidsList(profile.getDomain())
                b = allBids.get(randint(0, allBids.size() - 1))
                while profile.getUtility(b) < (1.0 - (0.4 * self._progress.get((int)(time.time() * 1000)))):
                    b = allBids.get(randint(0, allBids.size() - 1))
                bid = b
                return bid
            else:
                i = 0.0

        # find the coordinates of the new bid we are going to make
        x = profile.getUtility(self._last_sent_bid) + Decimal(i) * self._agent_util_change
        y = self._opponent_model.getUtility(self._last_sent_bid) + Decimal(i) * self._opponent_util_change

        # make sure the values of the coordinates are not below 0 or above 1
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        if y < 0:
            y = 0
        if y > 1:
            y = 1

        # find the most matching bid to the calculated coordinates
        for b in self._bids_and_utils.values():
            agent_util = profile.getUtility(b)
            opp_util = self._opponent_model.getUtility(b)
            if float(x) + 0.05 > agent_util > float(x) - 0.05:
                if float(y) + 0.05 > opp_util > float(y) - 0.05:
                    return b

        # if no bid was found generate a random bid that has a
        # utility above a certain value, dependent on how high the progress is
        allBids = AllBidsList(profile.getDomain())
        b = allBids.get(randint(0, allBids.size() - 1))
        while profile.getUtility(b) < profile.getUtility(self._last_sent_bid) or profile.getUtility(b) < (1.0 - (0.4 * self._progress.get((int)(time.time() * 1000)))):
            b = allBids.get(randint(0, allBids.size() - 1))
        bid = b
        return bid
