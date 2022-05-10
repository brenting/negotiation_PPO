import operator
import random
import time
from pathlib import Path
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
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.simplerunner.NegoRunner import StdOutReporter

from agent.utils.opponent_model import OpponentModel
from environment.negotiation import NegotiationEnv
from .utils.ppo import PPO

################ PPO hyperparameters ################
PPO_PARAMETERS = {
    "state_dim": 4,  # dimension of state space
    "action_dim": 2,  # dimension of action space
    "lr_actor": 0.0003,  # learning rate for actor network
    "lr_critic": 0.001,  # learning rate for critic network
    "gamma": 1,  # discount factor
    "K_epochs": 10,  # update policy for K epochs in one PPO update
    "eps_clip": 0.2,  # clip parameter for PPO
    "action_std": 0.6,  # starting std for action distribution (Multivariate Normal)
    "action_std_decay_rate": 0.05,  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    "min_action_std": 0.1,  # minimum action_std (stop decay after action_std <= min_action_std)
}
#####################################################

################## other parameters #################
LOG_FREQ = 100  # log avg reward in the interval (in num episode)
SAVE_MODEL_FREQ = 100  # save model frequency (in num episode)
ACTION_STD_DECAY_FREQ = 250  # action_std decay frequency (in num episode)
UPDATE_EPISODE_FREQ = 10  # update policy every n episodes


#####################################################


class AcceptanceAgent:
    def __init__(self, ppo: PPO = None):
        # create new PPO if none is provided
        if ppo is None:
            self.ppo = PPO(**PPO_PARAMETERS)
        elif isinstance(ppo, PPO):
            self.ppo: PPO = ppo
        else:
            raise ValueError(f"ppo argument is of wrong type: {type(ppo)}")

        self.domain: Domain = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.settings: Settings = None

        self.opp_concession = None
        self.received_utils = np.array([])
        self.reservation_bid_utility = 0
        self.bids_utility_map = {}
        self.max_possible_utility = 0
        self.all_bids = None
        self.acceptability_index = 0
        self.rewards = []

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
            self.received_utils = np.array([])
            self.reservation_bid_utility = 0
            self.bids_utility_map = {}
            self.max_possible_utility = 0
            self.acceptability_index = 0
            self.last_received_utils = [0.0, 0.0, 0.0]
            self.target = 1
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
            profile_connection.close()

            self.opponent_model = OpponentModel(self.domain)

            if self.profile.getReservationBid() is not None:
                self.reservation_bid_utility = self.profile.getUtility(
                    self.profile.getReservationBid())

            # Get list of all possible bids from domain
            self.all_bids = AllBidsList(self.profile.getDomain())

            # Filter all bids list & remove all bids with lower utility than reservation bid
            self.all_bids = filter(lambda x: self.profile.getUtility(x) >= self.reservation_bid_utility,
                                   self.all_bids)

            # Sort list of all bids descending by utility
            self.all_bids = sorted(self.all_bids, key=lambda x: self.profile.getUtility(x),
                                   reverse=True)

            # Get maximum possible utility
            self.max_possible_utility = float(self.profile.getUtility(self.all_bids[0]))

            # Create a map bid-utility
            for bid in self.all_bids:
                self.bids_utility_map[bid] = self.profile.getUtility(bid)
            self.bids_utility_map = dict(
                sorted(self.bids_utility_map.items(), key=operator.itemgetter(1), reverse=True))

    def select_action(self, obs: Offer) -> Action:
        """Method to return an action when it is our turn.

        Args:
            obs (Offer): Observation in the form of the offer made by the opponent.

        Returns:
            Action: Our action as a response on the Offer.
        """
        # read  opponent  concession factor from file
        # with open("evalue.txt", "r") as f:
        #     self.opp_concession = float(f.read())
        #     # print(self.opp_concession, type(self.opp_concession))

        # extract bid from offer and use it to update opponent utility estimation
        received_bid = obs.getBid() if obs is not None else None
        self.opponent_model.update(received_bid)
        opp_utility = self.opponent_model.get_predicted_utility(received_bid)
        # self.opponent_model.update(received_bid)
        self.received_utils = np.append(self.received_utils, [self.get_utility(received_bid)])

        # find a good bid based on the utility goals
        target = self.get_target_utility()
        self.target = self.target if target is None else target
        bid = self.find_bid()
        action = Offer(self.me, bid)

        # build state vector for PPO
        received_util = float(self.get_utility(received_bid))
        self.last_received_utils.append(received_util)
        self.last_received_utils.pop(0)
        progress = self.progress.get(time.time() * 1000)
        utility_of_next_bid = 0 if bid is None else self.get_utility(bid)
        state = self.last_received_utils + [progress]
        assert len(state) == PPO_PARAMETERS["state_dim"]

        # obtain action vector from PPo based on the state
        accept = self.ppo.select_action(state)
        assert len(accept) == PPO_PARAMETERS["action_dim"]

        # return Accept if the received offer is better than our goal
        if accept[0] > accept[1]:
            # print("accepted", target, utility_of_next_bid, received_util,)
            return Accept(self.me, received_bid)

        return action

    def find_bid(self) -> Bid:
        while self.bids_utility_map[
            self.all_bids[self.acceptability_index]] > self.target and self.acceptability_index < len(
                self.all_bids) - 1:
            self.acceptability_index = self.acceptability_index + 1

        acceptable = self.all_bids[0:self.acceptability_index]

        if len(acceptable) == 0:
            acceptable.append(list(self.bids_utility_map.keys())[0])

        random.shuffle(acceptable)
        bid = acceptable[0]
        # print(self.target, self.get_utility(bid))
        return bid

    def get_utility(self, bid: Bid) -> float:
        """returns utility value of bid for our agent"""
        return float(self.profile.getUtility(bid))

    def get_target_utility(self):
        return self.get_ratio() * self.get_initial_target_utility() + (1 - self.get_ratio())

    def get_initial_target_utility(self):
        progress = self.progress.get(time.time() * 1000)
        t = 1 - (1 - self.emax()) * (progress ** 7)
        return t

    def get_ratio(self):
        return np.min([(self.compute_width() + self.g()) / (1 - self.get_initial_target_utility()), 2])

    def g(self):
        return 0.5

    def emax(self):
        avg = self.compute_average()
        return avg * (1 - avg) * self.compute_width()

    def compute_average(self):
        return np.mean(self.received_utils)

    def compute_width(self):
        avg = self.compute_average()
        variance = np.abs(np.mean(self.received_utils ** 2 - avg ** 2))
        # print("variance:",variance)
        return np.sqrt(12 * variance)

    def save(self, checkpoint_path):
        self.ppo.save(checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path):
        ppo = PPO(**PPO_PARAMETERS)
        ppo.load(checkpoint_path)
        return cls(ppo)

    ########################################################################################
    ############################### training method of agent ###############################
    ########################################################################################
    def train(
            self, env: NegotiationEnv, time_budget_sec: int, checkpoint_path: str) -> list:
        log_dir_path = Path("logs")
        # create results directory if it does not exist
        if not log_dir_path.exists():
            log_dir_path.mkdir(parents=True)

        # track total training time
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = log_dir_path.joinpath(f"{start_time}.csv")

        print(f"Started training at (GMT) : {start_time}")
        print("=" * 100)

        # logging file
        log_f = open(log_file_path, "w+")
        log_f.write("episode,reward\n")

        # logging variables
        log_running_reward = []

        # training loop
        episode_count = 0
        start_time_sec = time.time()
        while (time.time() - start_time_sec) < time_budget_sec:

            obs = env.reset(self)
            episode_reward = 0
            done = False

            while not done:
                # select action with policy
                action = self.select_action(obs)
                # execute action
                obs, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                self.ppo.buffer.rewards.append(reward)
                self.ppo.buffer.is_terminals.append(done)

                episode_reward += reward
            log_running_reward.append(episode_reward)
            self.rewards.append(episode_reward)
            episode_count += 1
            #print(episode_count,self.rewards )

            # update PPO agent every n sessions
            if episode_count % UPDATE_EPISODE_FREQ == 0:
                self.ppo.update()

            # decay action std of ouput action distribution
            if episode_count % ACTION_STD_DECAY_FREQ == 0:
                self.ppo.decay_action_std()

            # log in logging file
            if episode_count % LOG_FREQ == 0:
                # log average reward since last log
                log_avg_reward = sum(log_running_reward) / len(log_running_reward)
                log_f.write(f"{episode_count},{log_avg_reward:.4f}\n")
                log_f.flush()
                log_running_reward = []

            # save model weights
            if episode_count % SAVE_MODEL_FREQ == 0:
                print("―" * 100)
                print(f"saving model at: {checkpoint_path}")
                self.ppo.save(checkpoint_path)
                print("model saved")
                print("―" * 100)

        log_f.close()
        env.close()

        # print total training time
        print("=" * 100)
        print(f"Total training time: {time.time() - start_time_sec}")
        print("Episode count:", episode_count)
        print("=" * 100)
        return log_running_reward
