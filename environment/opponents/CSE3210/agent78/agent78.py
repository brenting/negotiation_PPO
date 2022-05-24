import logging
from random import randint, random
import traceback
from typing import cast, Dict, List, Set, Collection

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.bidspace.BidsWithUtility import BidsWithUtility
from geniusweb.actions.LearningDone import LearningDone
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.actions.Vote import Vote
from geniusweb.actions.Votes import Votes
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.OptIn import OptIn
from geniusweb.inform.Settings import Settings
from geniusweb.inform.Voting import Voting
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
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
from geniusweb.utils import val
from geniusweb.profileconnection.ProfileInterface import ProfileInterface
from geniusweb.profile.utilityspace.LinearAdditive import LinearAdditive
from geniusweb.progress.Progress import Progress
from tudelft.utilities.immutablelist.ImmutableList import ImmutableList
from time import sleep, time as clock
from decimal import Decimal
import sys
from ...time_dependent_agent.extended_util_space import ExtendedUtilSpace
from tudelft_utilities_logging.Reporter import Reporter
from geniusweb.opponentmodel.FrequencyOpponentModel import FrequencyOpponentModel




class Agent78(DefaultParty):
    """
    Combination of Boulware with Frequency Opponent Modelling strategy
    """

    def __init__(self, reporter: Reporter = None):
        super().__init__(reporter)
        self._profileint: ProfileInterface = None  # type:ignore
        self._utilspace: LinearAdditive = None  # type:ignore
        self._me: PartyId = None  # type:ignore
        self._progress: Progress = None  # type:ignore
        self._lastReceivedBid: Bid = None  # type:ignore
        self._extendedspace: ExtendedUtilSpace = None  # type:ignore
        self._e: float = 0.1
        self._lastvotes: Votes = None  # type:ignore
        self._settings: Settings = None  # type:ignore
        self._fom: FrequencyOpponentModel = None # type:ignore
        self.getReporter().log(logging.INFO, "party is initialized")
        # self._bidutils = BidsWithUtility.create(self._utilspace)


    # Override
    def getCapabilities(self) -> Capabilities:
        return Capabilities(
            set(["SAOP", "Learn", "MOPAC"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    # Override
    def notifyChange(self, info: Inform):
        try:
            if isinstance(info, Settings):
                self._settings = info
                self._me = self._settings.getID()
                self._progress = self._settings.getProgress()

                self._profileint = ProfileConnectionFactory.create(
                    self._settings.getProfile().getURI(), self.getReporter()
                )
                self._fom = FrequencyOpponentModel.create()
                self._fom = self._fom.With(self._profileint.getProfile().getDomain(), None)

            elif isinstance(info, ActionDone):
                otheract: Action = info.getAction()
                if isinstance(otheract, Offer):
                    self._fom = self._fom.WithAction(otheract, self._progress)
                    self._lastReceivedBid = otheract.getBid()
            elif isinstance(info, YourTurn):
                self._delayResponse()
                return self._myTurn()
            elif isinstance(info, Finished):
                self.getReporter().log(logging.INFO, "Final ourcome:" + str(info))
                self.terminate()
                # stop this party and free resources.
        except Exception as ex:
            self.getReporter().log(logging.CRITICAL, "Failed to handle info", ex)
        self._updateRound(info)

    def getE(self) -> float:
        """
        @return the E value that controls the party's behaviour. Depending on the
                value of e, extreme sets show clearly different patterns of
               behaviour [1]:

               1. Boulware: For this strategy e &lt; 1 and the initial offer is
                maintained till time is almost exhausted, when the agent concedes
                up to its reservation value.

                2. Conceder: For this strategy e &gt; 1 and the agent goes to its
                reservation value very quickly.

                3. When e = 1, the price is increased linearly.

                4. When e = 0, the agent plays hardball.
        """
        return self._e

    # Override
    def getDescription(self) -> str:
        return ("Combination of Boulware with e=0.10  with BOA strategies")

    # Override
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        super().terminate()
        if self._profileint != None:
            self._profileint.close()
            self._profileint = None

    ##################### private support funcs #########################

    def _updateRound(self, info: Inform):
        """
        Update {@link #progress}, depending on the protocol and last received
        {@link Inform}

        @param info the received info.
        """
        if self._settings == None:  # not yet initialized
            return
        protocol: str = str(self._settings.getProtocol().getURI())

        if "SAOP" == protocol or "SHAOP" == protocol:
            if not isinstance(info, YourTurn):
                return
        else:
            return
        # if we get here, round must be increased.
        if isinstance(self._progress, ProgressRounds):
            self._progress = self._progress.advance()

    def _myTurn(self):
        self._updateUtilSpace()
        bid = self._makeBid()

        myAction: Action
        if bid == None or (
                self._lastReceivedBid != None
                and self._utilspace.getUtility(self._lastReceivedBid)
                >= self._utilspace.getUtility(bid)
        ):
            # if bid==null we failed to suggest next bid.
            myAction = Accept(self._me, self._lastReceivedBid)
        else:
            myAction = Offer(self._me, bid)
        return myAction

    def _updateUtilSpace(self) -> LinearAdditive:  # throws IOException
        newutilspace = self._profileint.getProfile()
        if not newutilspace == self._utilspace:
            self._utilspace = cast(LinearAdditive, newutilspace)
            self._extendedspace = ExtendedUtilSpace(self._utilspace)
        return self._utilspace

    def _makeBid(self) -> Bid:
        """
        @return next possible bid with current target utility, or null if no such
                bid.
        """
        time = self._progress.get(round(clock() * 1000))
        utilityGoal = self._getUtilityGoal(
            time,
            self.getE(),
            self._extendedspace.getMin(),
            self._extendedspace.getMax(),
        )
        # for utilityGoal higher than 0.95
        if utilityGoal > Decimal(0.95):
            utilityGoal = Decimal(1.0-((time * 200) % 5) / 100)

            options: ImmutableList[Bid] = self._extendedspace.getBids(utilityGoal)

            if options.size() == 0:
                # if we can't find good bid, get max util bid....
                options = self._extendedspace.getBids(self._extendedspace.getMax())
                return options.get(0)
            else:
                # return random bids when utilityGoal is still high to confuse opponent
                bid: Bid = self._randomBid(options)
                return bid

        options: ImmutableList[Bid] = self._extendedspace.getBids(utilityGoal)
        if options.size() == 0:
            # if we can't find good bid, get max util bid....
            options = self._extendedspace.getBids(self._extendedspace.getMax())
            return options.get(0)
        # # pick the one with best nash product utility
        else:
            bid: Bid = self._bestBid(options)
            return bid

    def _bestBid(self, options: ImmutableList[Bid]) -> Bid:
        """
        @return bid with best nash product according to the utility of the opponent
        estimated by the frequency opponent model.
        """
        maxUtil = Decimal(0)
        maxBid = None
        profile = self._profileint.getProfile()
        for bid in options:
            opponentUtility = max(self._fom.getUtility(bid), Decimal(0.001))
            ownUtility = profile.getUtility(bid)
            nash_product = Decimal(opponentUtility * ownUtility)  # nash product
            if(nash_product > maxUtil):
                maxUtil = nash_product
                maxBid = bid
        return maxBid  # return bid with highest nash product

    def _randomBid(self, options: ImmutableList[Bid]) -> Bid:
        """
        @return random bid from list of bids.
        """
        r = randint(0, options.size() -1)
        return options.get(r)

    def _getUtilityGoal(
            self, t: float, e: float, minUtil: Decimal, maxUtil: Decimal
    ) -> Decimal:
        """
        @param t       the time in [0,1] where 0 means start of nego and 1 the
                       end of nego (absolute time/round limit)
        @param e       the e value that determinses how fast the party makes
                       concessions with time. Typically around 1. 0 means no
                       concession, 1 linear concession, &gt;1 faster than linear
                       concession.
        @param minUtil the minimum utility possible in our profile
        @param maxUtil the maximum utility possible in our profile
        @return the utility goal for this time and e value
        """

        ft1 = Decimal(1)
        if e != 0:
            ft1 = round(Decimal(1 - pow(t, 1 / e)), 6)  # defaults ROUND_HALF_UP
        return max(min((minUtil + (maxUtil - minUtil) * ft1), maxUtil), minUtil)


    def _isGood(self, bid: Bid) -> bool:
        """
        @param bid the bid to check
        @return true iff bid is good for us.
        """
        if bid == None or self._profileint == None:
            return False
        profile = cast(LinearAdditive, self._profileint.getProfile())
        # the profile MUST contain UtilitySpace
        time = self._progress.get(round(clock() * 1000))
        return profile.getUtility(bid) >= self._getUtilityGoal(
            time,
            self.getE(),
            self._extendedspace.getMin(),
            self._extendedspace.getMax(),
        )

    def _delayResponse(self):  # throws InterruptedException
        """
        Do random delay of provided delay in seconds, randomized by factor in
        [0.5, 1.5]. Does not delay if set to 0.

        @throws InterruptedException
        """
        delay = self._settings.getParameters().getDouble("delay", 0, 0, 10000000)
        if delay > 0:
            sleep(delay * (0.5 + random()))
