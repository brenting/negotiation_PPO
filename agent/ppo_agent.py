import time
from pathlib import Path
from random import randint
from typing import cast

import numpy as np
from environment.negotiation import NegotiationEnv
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
from geniusweb.references.Parameters import Parameters
from geniusweb.simplerunner.NegoRunner import StdOutReporter

from agent.utils.opponent_model import OpponentModel

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


class PPOAgent:
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

        self.opponent_model: OpponentModel = None

        self.last_received_utils = [0.0, 0.0, 0.0]

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
            profile_connection.close()

            self.opponent_model = OpponentModel(self.domain)

    def select_action(self, obs: Offer) -> Action:
        """Method to return an action when it is our turn.

        Args:
            obs (Offer): Observation in the form of the offer made by the opponent.

        Returns:
            Action: Our action as a response on the Offer.
        """
        # extract bid from offer and use it to update opponent utility estimation
        received_bid = obs.getBid()
        self.opponent_model.update(received_bid)

        # build state vector for PPO
        received_util = float(self.get_utility(received_bid))
        self.last_received_utils.append(received_util)
        self.last_received_utils.pop(0)
        progress = self.progress.get(time.time() * 1000)
        state = tuple(self.last_received_utils + [progress])
        assert len(state) == PPO_PARAMETERS["state_dim"]

        # obtain action vector from PPo based on the state
        util_goals = self.ppo.select_action(state)
        assert len(util_goals) == PPO_PARAMETERS["action_dim"]

        # return Accept if the received offer is better than our goal
        if util_goals[0] < received_util:
            return Accept(self.me, received_bid)

        # find a good bid based on the utility goals
        bid = self.find_bid(util_goals)
        action = Offer(self.me, bid)

        return action

    def find_bid(self, util_goals: np.ndarray) -> Bid:
        # TODO: Implement NSGA-II version?
        # compose a list of all possible bids
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        best_difference = 99999.0
        best_bid = None

        # take 1000 random attempts to find a bid close to both utility goals
        for _ in range(1000):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            my_util = self.get_utility(bid)
            opp_util = self.opponent_model.get_predicted_utility(bid)
            difference = np.sum(np.square(util_goals - np.array([my_util, opp_util])))
            if difference < best_difference and my_util > util_goals[0]:
                best_difference, best_bid = difference, bid

        return best_bid

    def get_utility(self, bid: Bid) -> float:
        """returns utility value of bid for our agent"""
        return float(self.profile.getUtility(bid))

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
        self, env: NegotiationEnv, time_budget_sec: int, checkpoint_path: str
    ) -> None:
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
            episode_count += 1

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
        print("=" * 100)
