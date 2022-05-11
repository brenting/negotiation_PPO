import logging
import time
from random import randint
from typing import cast

import numpy as np
from . import agentUtils
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
import tempfile


class Agent70(DefaultParty):
    """
    A One-shot negotiating agent, who will make selfish bids until the last 10 rounds,
    at which point it will use the information about the opponent to make bids that are agreeable to both parties
    """

    def __init__(self, reporter: Reporter = None):
        super().__init__(reporter)
        self.getReporter().log(logging.INFO, "party is initialized")
        self._profile = None
        self._last_received_bid: Bid = None

        self.minimumUtil = 0.85             #The minimum utility score that our agent will accept
        self.reservation = None             #The utility of our reservation (or backup) bid, our minimumUtil should never be lower than this
        self.bestUtil = 0.0                 #The utility score of the best bid the opponent has offered
        self.bestBid: Bid = None            #The best bid offered by the opponent

        self.previousBid: Bid = None        #The last bid offered by the opponent
        self.opponentSanity = 1             #A measure of how coherent the opponents bids are, the lower the sanity, the higher chance the opponent is a random walker
        self.opponentProfile = None         #A dictionary keeping track of how often each value has been chosen for each bid
        self.BOD = 10                       #The minimum number of times an opponent needs to make an "insane" bid for them to be marked as a random walker
        self.sanityThresh = 0.4             #How much the opponent's current bid needs to differ from their previous bid for it to be marked "insane"
        self.sane = True                    #Whether the opponent is a random walker
        self.canPrint = False               #Debugging toggle
        self.smartBidStart = 0.95
        self.aggression = 0.3               #a factor determining how much the minimumUtil value decreases to match the opponents best bid each round once the agent starts making final bids
        # self.tempDir = f"{ tempfile.gettempdir()}/Group70_Negotiation_assignment/file.txt"    #file path for temporary storage
        # self.tempDir = f"Group70_Negotiation_assignment/file.txt"    #file path for temporary storage
        self.file = None

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
            self._progress = self._settings.getProgress()

            # the profile contains the preferences of the agent over the domain
            self._profile = ProfileConnectionFactory.create(
                info.getProfile().getURI(), self.getReporter()
            )
            self.opponentProfile = agentUtils.initProfileDict(self)
            self.reservation = self._profile.getProfile().getReservationBid()
            if self.reservation is None: self.reservation = 0
            # self.file = agentUtils.loadFile(self)
            # newSBS = agentUtils.recalculateSBS(self, agentUtils.parseFile(self))
            # if(newSBS > 0):
            #     self.smartBidStart = newSBS
            # print("!!!TEST AGENT INITIALISED!!!")
            # print(F"!!!NEW VALUE OF SBS={self.smartBidStart}!!!")
            # print("!!!TEST AGENT INITIALISED!!!")
        # ActionDone is an action send by an opponent (an offer or an accept)
        elif isinstance(info, ActionDone):
            action: Action = cast(ActionDone, info).getAction()

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
            # agentUtils.writeFile(self)
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
        return "test agent, i dont know what it does!"

    # execute a turn
    def _myTurn(self):
        # check if the last received offer if the opponent is good enough
        if self._isGood(self._last_received_bid):
            # if so, accept the offer
            if self.canPrint: print("ACCEPTING")
            action = Accept(self._me, self._last_received_bid)
        else:       #if the bid is not accepted, the opponents profile is first updated, as well as checks to see if they are a random walker

            if self.getUtil(self._last_received_bid) > self.bestUtil:       #comparing the util of the opponents best bid the the previously recorded one
                self.bestUtil = float(self.getUtil(self._last_received_bid))#updating the best util
                self.bestBid =self._last_received_bid                           #updating the best bid made by the opponent

            agentUtils.updateProfile(self, self._last_received_bid)         #adds the opponents bid to the dictionary recording the frequencies of their bids

            if self._last_received_bid != None:                             #if the received bid is valid:
                totalSanity = agentUtils.getSanity(self, self._last_received_bid)   #calculate how cohesive the opponents most recent bid, compared to all of their previous bids
                currentSanity = 1                                                   
                if(self.previousBid != None):                                       #if this is not the opponents first bid:
                    currentSanity = agentUtils.compareBids(self.previousBid, self._last_received_bid)   #check if the opponents bid is "sane" when compared to their last one
                    if self.sane:                                                                       #if the opponent hasnt already been marked as insane
                        if currentSanity < self.sanityThresh:                                           #if the sanity of the opponents last bid is not sufficiently sane
                            self.BOD -= 1                                                               #decrement the BOD
                            if self.BOD == 0:                                                           #if the BOD == 0, the opponent is marked as "insane" and it is assumed that their bids are random
                                if self.canPrint: print("YOU ARE INSANE YOU ARE NOT REAL I AM IN YOUR WALLS YOU ARE INSANE")
                                if self.canPrint: print("YOU ARE INSANE YOU ARE NOT REAL I AM IN YOUR WALLS YOU ARE INSANE")
                                if self.canPrint: print("YOU ARE INSANE YOU ARE NOT REAL I AM IN YOUR WALLS YOU ARE INSANE")
                                if self.canPrint: print("YOU ARE INSANE YOU ARE NOT REAL I AM IN YOUR WALLS YOU ARE INSANE")
                                self.sane = False
                        else:
                            self.BOD += 1                                                               #if the sanity of the opponents last bid was sufficient, increment the BOD
                if self.canPrint: print(f"progress={self._progress.get(time.time() * 1000)}, totalSanity={totalSanity}, currentSanity={currentSanity}, BOD={self.BOD}")
                if self.canPrint: print(f"minimumUtil={self.minimumUtil}")
                self.previousBid = self._last_received_bid                  #update the previous bid
            # then, find a bid to propose as counter offer
            if(self._progress.get(time.time() * 1000) >= 0.99):                  #if it is the last (or second last) opportunity to bid, then offer the best bid recieved from the opponent
                bid = self.bestBid                              #it is assumed that the opponent will always accept a bid they have previously made, so this garuntees an agreement.
            
            elif(self.sane and self._progress.get(time.time() * 1000) >= self.smartBidStart):   #if it is nearing the end of bidding, then start trying to make bids that appease both parties
                print(f"finding good bid with minimumUtil={self.minimumUtil}")
                bid = self.constructGoodBid()                        #findGoodBid constructs a bid that both parties will find favourable

                self.minimumUtil -= ((self.minimumUtil - self.bestUtil)*(self.aggression))    #in the case that this bid is rejected, and bidding continues, lower the minimum acceptable utility score
            else:
                bid = self._findBid()                           #if bidding is not close to ending, make a selfish bid with a high utility score

            if self.canPrint: 
                print(f"bid: {bid}")
                util = self._profile.getProfile().getUtility(bid)
                print(f"  with util={util}")
                print(agentUtils.oppProfileToString(self))

            action = Offer(self._me, bid)                       #make offer using chosen bid
        # send the action
        return action

    # method that checks if we would agree with an offer
    def _isGood(self, bid: Bid) -> bool:
        # print(bid)
        # print(self.reservation)
        if bid is None:
            return False
        profile = self._profile.getProfile()

        # simply checks if the bid has a utility above our minimum acceptable score.
        #it does not matter how much time has passed; if the offer is good there is no reason to reject it
        #WAIT MAYBE THERE IS LOL
        return profile.getUtility(bid) > max(self.minimumUtil, self.reservation)


    #the bidding method used when the negotiation is not close to ending
    #chooses a random bid with a high utility score
    #these bids are not expected to be accepted, so there is no need to lower our standards for our opponent.
    def _findBid(self) -> Bid:
        # compose a list of all possible bids
        domain = self._profile.getProfile().getDomain()
        all_bids = AllBidsList(domain)
        profile = self._profile.getProfile()
        bBid = None
        bUtil = 0
        # make attempts at finding a random bid until one is found that is acceptable to us
        while bUtil < self.minimumUtil:
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            util = profile.getUtility(bid)
            bUtil = util
            bBid = bid
        return bBid

    #A method that uses the opponent profile that has been created during negotiations to make bid that is favourable to both
    def constructGoodBid(self) -> Bid:
        # print("constructing good bid")
        issues, mats = agentUtils.makeMoveMats(self)                #retrieves a list of issues, as well as a matrix of values and how frequently they were chosen for each value
        indexes = [[0 for _ in issues]]                     #indexes stores arrays of intergers, with each value representing the index of its corresponding value
                                                            #for example indexes[n][5]=2 means that for attempt n, we will use the second most frequently picked value for issue 5.
                                                            #the first array in indexes will be all 0s, meaning that the most frequently picked value for each issue will be used
        bid = self.createBid(issues, indexes[0], mats)      #create a bid using the opponents most frequently picked values for each issue
        if(self._isGood(bid)): return bid            #if that bid satisfies our requirements, return the bid
        i = 0
        for num in range (1, pow(2, len(issues))):          #otherwise, start iterating through every possible combination of values
            # print(f"num={num}")
            indexes = agentUtils.expandIndexes(indexes, num)            #add several new index arrays to indexes, starting with the next most commonly used issues
            # print("INDEXES AFTER EXPANSION: ", indexes)
            # print(f"INDEXES[{1}] AFTER EXPANSION: ", indexes[1])
            while(i < len(indexes)):                                    #for each of these newly added index combinations,
                bid = self.createBid(issues, indexes[i], mats)              #create a new bid using the next set of indexes
                if(self._isGood(bid)): return bid                    #if the bid meets our requirements, return bid
                i+=1
        
        return self._findBid()      #if none of those combinations are satisfactory just bid using our regular strategy



    def createBid(self, issues, indexes, mats) -> Bid:
        bidDict = dict()                                    #creates the dictionary which contains the bids
        # print(indexes)
        # print(indexes[0])
        for i, issue in enumerate(issues):                  #iterates through all of the issues and adds the value at the given index
            values = mats[i]
            index = min(indexes[i], len(values)-1)          #this is just to ensure that we dont go out of bounds.
            bidDict[issue] = values[index][1]               #adds the selected value to the dictionary
        return Bid(bidDict)


    def getUtil(self, bid:Bid):
        if self.canPrint: print(f"getting util of bid: {bid}")
        if bid == None: return 0
        return self._profile.getProfile().getUtility(bid)
