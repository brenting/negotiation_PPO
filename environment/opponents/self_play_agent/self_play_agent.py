import logging
import traceback
import time
from pathlib import Path
from random import randint
from typing import Collection, Dict, List, Set, cast

import numpy as np
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.EndNegotiation import EndNegotiation
from geniusweb.actions.LearningDone import LearningDone
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.actions.Vote import Vote
from geniusweb.actions.Votes import Votes
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Agreements import Agreements
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
from geniusweb.profile.utilityspace import LinearAdditiveUtilitySpace
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profileconnection.ProfileConnectionFactory import \
    ProfileConnectionFactory
from geniusweb.progress.ProgressRounds import ProgressRounds
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.utils import val
from tudelft_utilities_logging.Reporter import Reporter

from agent.utils.opponent_model import OpponentModel
from agent.utils.ppo import PPO
from environment.negotiation import NegotiationEnv

PPO_PARAMETERS = {
    "state_dim": 4,  # dimension of state space
    "action_dim": 2,  # dimension of action space
    "lr_actor": 0.0003,  # learning rate for actor network
    "lr_critic": 0.001,  # learning rate for critic network
    "gamma": 1,  # discount factor
    "K_epochs": 3,  # update policy for K epochs in one PPO update
    "eps_clip": 0.2,  # clip parameter for PPO
    "action_std": 0.6,  # starting std for action distribution (Multivariate Normal)
    "action_std_decay_rate": 0.05,  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    "min_action_std": 0.1,  # minimum action_std (stop decay after action_std <= min_action_std)
}


# LOG_FREQ = 10  # log avg reward in the interval (in num episode)
# SAVE_MODEL_FREQ = 10  # save model frequency (in num episode)
# ACTION_STD_DECAY_FREQ = 25  # action_std decay frequency (in num episode)
# UPDATE_EPISODE_FREQ = 10  # update policy every n episodes
# NUMBER_OF_AGENTS = 10

LOG_FREQ = 100  # log avg reward in the interval (in num episode)
SAVE_MODEL_FREQ = 100  # save model frequency (in num episode)
ACTION_STD_DECAY_FREQ = 40  # action_std decay frequency (in num episode)
UPDATE_EPISODE_FREQ = 20  # update policy every n episodes
NUMBER_OF_AGENTS = 5


class SelfPlayAgent(DefaultParty):
    """

    """

    ppo = []

    negotiation_count = np.zeros(NUMBER_OF_AGENTS)

    index = 0

    # last_received_utils = []

    # log_dir_path = Path("logs-2")
    # # create results directory if it does not exist
    # if not log_dir_path.exists():
    #     log_dir_path.mkdir(parents=True)
    #
    # # track total training time
    # start_time = time.strftime("%Y-%m-%d_%H%M%S")
    # log_file_path = log_dir_path.joinpath(f"{start_time}.csv")
    #
    # print(f"Started training-2 at (GMT) : {start_time}")
    # print("=" * 100)
    #
    # # logging file
    # log_f = open(log_file_path, "w+")
    # log_f.write("episode,reward\n")
    #
    # # logging variables
    # log_running_reward = []
    # episode_reward = 0.0

    def __init__(self, reporter: Reporter = None):
        super().__init__(reporter)
        self.getReporter().log(logging.INFO, "party is initialized")
        self.profile = None
        self.protocol = None
        self.lastOffer: Offer = None

        while len(SelfPlayAgent.ppo) < NUMBER_OF_AGENTS:
            print("Resetting PPO")
            SelfPlayAgent.ppo.append(PPO(**PPO_PARAMETERS))
            # SelfPlayAgent.last_received_utils.append([0.0, 0.0, 0.0])

        self.last_received_utils = [0.0, 0.0, 0.0]
        self.domain: Domain = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.settings: Settings = None
        self.all_bids = None

        self.opponent_model: OpponentModel = None

    # Override
    def notifyChange(self, info: Inform):
        # self.getReporter().log(logging.INFO,"received info:"+str(info))
        if isinstance(info, Settings):
            self.settings: Settings = cast(Settings, info)
            self.me = self.settings.getID()
            self.protocol: str = str(self.settings.getProtocol().getURI())
            self.progress = self.settings.getProgress()
            if len(SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.rewards) != len(SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.states):
                SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.rewards.append(0.0)
                SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.is_terminals.append(True)
            if "Learn" == self.protocol:
                return LearningDone(self._me)  # type:ignore
            else:
                self.profile = ProfileConnectionFactory.create(
                    info.getProfile().getURI(), self.getReporter()
                )
            SelfPlayAgent.index = randint(0, NUMBER_OF_AGENTS - 1)

            # if SelfPlayAgent.ppo is None:
            #     print("Resetting PPO")
            #     SelfPlayAgent.ppo = PPO(**PPO_PARAMETERS)
            # else:
            #     SelfPlayAgent.log_running_reward.append(SelfPlayAgent.episode_reward)
            #     SelfPlayAgent.episode_reward = 0.0
            self.last_received_utils = [0.0, 0.0, 0.0]
            self.lastOffer = None
            self.domain = self.profile.getProfile().getDomain()
            self.all_bids = AllBidsList(self.domain)
            self.opponent_model = OpponentModel(self.domain)
            SelfPlayAgent.negotiation_count[SelfPlayAgent.index] += 1
            if SelfPlayAgent.negotiation_count[SelfPlayAgent.index] % UPDATE_EPISODE_FREQ == 0:
                SelfPlayAgent.ppo[SelfPlayAgent.index].update()

            if SelfPlayAgent.negotiation_count[SelfPlayAgent.index] % ACTION_STD_DECAY_FREQ == 0:
                SelfPlayAgent.ppo[SelfPlayAgent.index].decay_action_std()

            # if SelfPlayAgent.negotiation_count % LOG_FREQ == 0:
                # log average reward since last log
                # print(sum(SelfPlayAgent.log_running_reward))
                # print(len(SelfPlayAgent.log_running_reward))
                # print(SelfPlayAgent.log_running_reward)
                # log_avg_reward = sum(SelfPlayAgent.log_running_reward) / len(SelfPlayAgent.log_running_reward)
                # SelfPlayAgent.log_f.write(f"{SelfPlayAgent.negotiation_count},{log_avg_reward:.4f}\n")
                # SelfPlayAgent.log_f.flush()
                # SelfPlayAgent.log_running_reward = []

        elif isinstance(info, ActionDone):
            action = cast(ActionDone, info).getAction()
            # if isinstance(action, EndNegotiation):
            #     SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.rewards.append(0.0)
            #     SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.is_terminals.append(True)
            if isinstance(action, Offer):
                self.lastOffer = cast(Offer, action)
                SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.rewards.append(0.0)
                SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.is_terminals.append(False)
            if isinstance(action, Accept):
                SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.rewards.append(self.get_utility(action.getBid()))
                SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.is_terminals.append(True)
            # if isinstance(action, Accept):
                # if action.getBid() is not None:
                #     SelfPlayAgent.episode_reward += self.get_utility(action.getBid())
                    # print("Received bid with utility: " + str(self.get_utility(action.getBid())))
        elif isinstance(info, YourTurn):
            action = self._myTurn()
            if isinstance(self.progress, ProgressRounds):
                self.progress = self.progress.advance()
            if isinstance(action, Accept):
                SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.rewards.append(self.get_utility(action.getBid()))
                SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.is_terminals.append(True)
            # elif isinstance(action, Offer):
                # SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.rewards.append(0.0)
                # SelfPlayAgent.ppo[SelfPlayAgent.index].buffer.is_terminals.append(False)
            return action
        elif isinstance(info, Finished):
            self.terminate()
        else:
            self.getReporter().log(
                logging.WARNING, "Ignoring unknown info " + str(info)
            )

    # Override
    def getCapabilities(self) -> Capabilities:
        return Capabilities(
            {"SAOP", "Learn", "MOPAC"},
            {"geniusweb.profile.utilityspace.LinearAdditive"},
        )

    # Override
    def getDescription(self) -> str:
        return "Offers random bids until a bid with sufficient utility is offered. Parameters minPower and maxPower can be used to control voting behaviour."

    # Override
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        super().terminate()
        if self.profile is not None:
            self.profile.close()
            self.profile = None

    def _myTurn(self):
        return self.select_action(self.lastOffer)

    def select_action(self, obs: Offer) -> Action:
        """Method to return an action when it is our turn.

        Args:
            obs (Offer): Observation in the form of the offer made by the opponent.

        Returns:
            Action: Our action as a response on the Offer.
        """
        # extract bid from offer and use it to update opponent utility estimation
        # print("Agent 2 received offer: " + str(obs))
        if obs is None or obs.getBid() is None:
            received_bid = None
            received_util = 0.0
        else:
            received_bid = obs.getBid()
            self.opponent_model.update(received_bid)
            received_util = float(self.get_utility(received_bid))

        # if received_bid is not None:
        #     SelfPlayAgent.episode_reward += 0.0
            # SelfPlayAgent.log_running_reward.append(0.0)
            # print("Received bid with utility: " + str(0.0))

        # build state vector for PPO
        self.last_received_utils.append(received_util)
        self.last_received_utils.pop(0)
        progress = self.progress.get(time.time() * 1000)
        state = tuple(self.last_received_utils + [progress])
        # state = tuple(self.last_received_utils[SelfPlayAgent.index] + [progress])
        assert len(state) == PPO_PARAMETERS["state_dim"]

        # obtain action vector from PPo based on the state
        util_goals = self.ppo[SelfPlayAgent.index].select_action(state)
        assert len(util_goals) == PPO_PARAMETERS["action_dim"]

        # return Accept if the reveived offer is better than our goal
        if util_goals[0] < received_util and received_bid is not None:
            # SelfPlayAgent.episode_reward += self.get_utility(received_bid)
            # SelfPlayAgent.log_running_reward.append(self.get_utility(received_bid))
            # print("Received bid with utility: " + str(received_util))
            return Accept(self.me, received_bid)

        # find a good bid based on the utility goals
        bid = self.find_bid(util_goals)
        action = Offer(self.me, bid)
        # print("Agent 2 offering: " + str(action))

        return action

    def find_bid(self, util_goals: np.ndarray) -> Bid:
        # TODO: Implement NSGA-II version?
        # compose a list of all possible bids
        best_difference = 99999.0
        best_bid = Bid({})

        # take 1000 random attempts to find a bid close to both utility goals
        for _ in range(1000):
            bid = self.all_bids.get(randint(0, self.all_bids.size() - 1))
            my_util = self.get_utility(bid)
            opp_util = self.opponent_model.get_predicted_utility(bid)
            difference = np.sum(np.square(util_goals - np.array([my_util, opp_util])))
            if difference < best_difference and my_util > util_goals[0]:
                best_difference, best_bid = difference, bid

        return best_bid

    def get_utility(self, bid: Bid) -> float:
        """returns utility value of bid for our agent"""
        return float(self.profile.getProfile().getUtility(bid))
