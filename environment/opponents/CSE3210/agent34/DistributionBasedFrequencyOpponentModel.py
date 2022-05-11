from decimal import Decimal
from math import isclose
from typing import Optional, Dict, List

from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.opponentmodel.OpponentModel import OpponentModel
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.progress.Progress import Progress
from geniusweb.references.Parameters import Parameters
from geniusweb.utils import val, HASH, toStr
from scipy.stats import chi2


def get_issue_max_count(freqs: Dict[Value, int]) -> int:
    """
    Method returns the maximum (Laplacian smoothed) number of occurrences of an issue value

    :param freqs: dictionary with value as key and number of occurrences as values.
    """
    # find max number of occurrences of an issue value
    max_count = 0
    for value, count in freqs.items():
        if count > max_count:
            max_count = count
    return max_count + 1  # Laplace smoothing


def update_rule(progress: float, alpha: float = 0.1, beta: float = 5):
    return alpha * (1 - progress ** beta)


def get_value_val(value: Value, freqs: Dict[Value, int]) -> float:
    """
    Method returns the estimated value of an issue value using a (Laplacian smoothed) frequency count

    :param value: issue Value that needs to be estimated
    :param freqs: dictionary with value as key and number of occurrences as values.
    """
    # find max number of occurrences of an issue value
    max_count = get_issue_max_count(freqs)

    count = 0
    if value in freqs:
        count = freqs.get(value)
    return (count + 1) / max_count  # Laplace smoothing





class DistributionBasedFrequencyOpponentModel(UtilitySpace, OpponentModel):

    _DECIMALS = 4

    """
    Opponent modelling implemented based on:

    Tunalı O., Aydoğan R., Sanchez-Anguix V. (2017) Rethinking Frequency Opponent Modeling in Automated Negotiation.
    In: An B., Bazzan A., Leite J., Villata S., van der Torre L. (eds) PRIMA 2017:
        Principles and Practice of Multi-Agent Systems. PRIMA 2017. Lecture Notes in Computer Science, vol 10621.
    Springer, Cham. https://doi.org/10.1007/978-3-319-69131-2_16

    This algorithm is an improvement on the classical frequency opponent modelling.
    The main differences lay in:
        1. the use of windows of offers instead of consecutive pairs of offers to offer a better estimate of
    the opponent's  issues' weights (in this case, issues' utilities)
        2. decayed weight updates
        3. slow growth of issue values importance
        4. weaker assumptions on the opponent's behaviour
    """

    def chi2_squared_test(self, observed: List[float], expected: List[float]) -> float:
        """
        Method returns the chi squared test for two probability distributions

        :param observed: observed frequency
        :param expected: expected frequency
        """
        sum1 = sum(observed)
        sum2 = sum(expected)
        if not isclose(sum1, sum2, rel_tol=1e-05) or len(observed) != len(expected):

            raise ValueError("Lists should have the same length and sum of elements")
        x2 = 0  # chi squared test
        for i in range(0, len(observed)):
            x2 += ((observed[i] - expected[i]) ** 2) / (expected[i] ** 2)
            # divide by expected value squared because passed arguments have < 1 values
        return x2

    def __init__(self,
                 finished_first_window: bool,
                 domain: Optional[Domain],
                 issue_weights: Dict[str, float],
                 prev_window: Dict[str, Dict[Value, int]],
                 current_window: Dict[str, Dict[Value, int]],
                 freqs: Dict[str, Dict[Value, int]],
                 cw_bids_count: int,
                 resBid: Optional[Bid], window_size: int,
                 gamma: float = 0.25, alpha: float = 0.1, beta: float = 5):

        """
        :param finished_first_window: whether negotiation has finished the first window of rounds
        :param domain
        :param issue_weights: weights of issues
        :param prev_window: dictionary that counts the frequencies of each value per issue in previous round
        :param current_window
        :param freqs: dictionary that counts the frequencies of each value per issue since start of negotiation
        :param cw_bids_count: current window bids count
        :param resBid: reservation bid
        :param window_size: window size, should be in (1, negotiation duration)
        :param gamma: value used for calculating the issue value valuation
        :param alpha: value used for updating weights
        :param beta: controls the decay in updating weights
        """
        self._finished_first_window = finished_first_window
        self._domain = domain
        self._issue_weights = issue_weights
        self._window_size = window_size

        if domain is not None:
            # initialize weights
            if not issue_weights:  # check that the weights dict is empty
                num_issues = len(domain.getIssues())
                # all issues have equal weight
                for issue in domain.getIssues():
                    issue_weights[issue] = 1 / num_issues
                # TODO: change this if you wanna change alpha
            # init window size
            # if negotiation_rounds != -1:
            #     self._window_size = floor(negotiation_rounds * window_fraction)

        self._prev_window = prev_window
        self._current_window = current_window
        self._bidFrequencies = freqs

        self._cw_bids_count = cw_bids_count
        self._resBid = resBid
        self._gamma = gamma
        self._alpha = alpha
        self._beta = beta

    @staticmethod
    def create(window_size, gamma: float = 0.25, alpha: float = 0.1, beta: float = 3) \
            -> "DistributionBasedFrequencyOpponentModel":
        """
        Method creates a DistributionBasedFrequencyOpponentModel with the passed params.

        :param window_size: size of window
        :param gamma: value used for calculating the issue value valuation
        :param alpha: value used for updating weights
        :param beta: controls the decay in updating weights
        """
        return DistributionBasedFrequencyOpponentModel(False, None, {}, {}, {}, {}, 0, None,
                                                       window_size,
                                                       gamma=gamma, alpha=alpha, beta=beta)

    def With(self, newDomain: Domain, newResBid: Optional[Bid]) -> "DistributionBasedFrequencyOpponentModel":
        if newDomain is None:
            raise ValueError("domain is not initialized")
        return DistributionBasedFrequencyOpponentModel(self._finished_first_window,
                                                       newDomain,
                                                       {},  # issue weights
                                                       {iss: {} for iss in newDomain.getIssues()},  # prev window
                                                       {iss: {} for iss in newDomain.getIssues()},  # curr window
                                                       {iss: {} for iss in newDomain.getIssues()},  # bid frequencies
                                                       0,  # curr window bids count
                                                       newResBid, self._window_size, gamma=self._gamma,
                                                       alpha=self._alpha, beta=self._beta)

    def getUtility(self, bid: Bid) -> Decimal:
        if self._domain is None:
            raise ValueError("domain is not initialized")
        utility = 0

        if bid is None:
            return Decimal(1)
        for issue in val(self._domain).getIssues():
            if issue in bid.getIssues():
                value = val(bid.getValue(issue))
                utility += get_value_val(value, self._bidFrequencies[issue]) * self._issue_weights.get(issue)
        return Decimal(round(utility, DistributionBasedFrequencyOpponentModel._DECIMALS))

    def getIssueWeights(self) -> Dict[str, float]:
        return self._issue_weights

    def getName(self) -> str:
        if self._domain is None:
            raise ValueError("domain is not initialized")
        return "DistributionBasedFreqOppModel" + str(hash(self)) + "For" + str(self._domain)

    def getDomain(self) -> Domain:
        return val(self._domain)

    # Override
    def WithParameters(self, parameters: Parameters) -> OpponentModel:
        return self  # ignore parameters

    def WithAction(self, action: Action, progress: Progress) -> "DistributionBasedFrequencyOpponentModel":
        if self._domain is None:
            raise ValueError("domain is not initialized")

        if not isinstance(action, Offer):
            return self

        bid: Bid = action.getBid()
        self._current_window, self._bidFrequencies = self._add_bid(bid)
        # check if current window is full
        if self._cw_bids_count == self._window_size:
            if not self._finished_first_window:
                return DistributionBasedFrequencyOpponentModel(True, self._domain, self._issue_weights, self._current_window,
                                                               {iss: {} for iss in self._domain.getIssues()},
                                                               self._bidFrequencies, 0, self._resBid,
                                                               window_size=self._window_size, gamma=self._gamma,
                                                               alpha=self._alpha,
                                                               beta=self._beta)
            # update weights and windows
            e: List[str] = list()
            concession = False

            for issue in self._domain.getIssues():
                prev_window_freqs = self._get_freq(self._prev_window[issue], issue)
                curr_window_freqs = self._get_freq(self._current_window[issue], issue)

                # use p_value of Chi Square test to determine if the distribution of issue values for 'issue' has
                # changed from the previous window of offers to the current one
                x2 = self.chi2_squared_test(prev_window_freqs, curr_window_freqs)
                p_val = chi2.sf(x2, 1)  # one degree of freedom

                # null hypothesis cannot be rejected
                # aka: opponent has not changed their behaviour
                if p_val > 0.05:
                    e.append(issue)
                else:
                    # null hypothesis is rejected
                    # check if opponent has conceded in the issue
                    valuation = self._get_valuation(self._bidFrequencies[issue], issue, gamma=self._gamma)
                    prev_window_utility = sum([x * y for x, y in zip(prev_window_freqs, valuation)])  # dot product
                    curr_window_utility = sum([x * y for x, y in zip(curr_window_freqs, valuation)])

                    # opponent has conceded
                    if curr_window_utility < prev_window_utility:
                        concession = True
            # update weights
            if len(e) != len(self._domain.getIssues()) and concession:
                for issue in e:
                    # update weights whose value distribution did not change over consecutive windows
                    # value distribution did not change => issue has greater weight
                    self._issue_weights[issue] = self._issue_weights[issue] + update_rule(progress.get(0),
                                                                                          alpha=self._alpha,
                                                                                          beta=self._beta)
                # normalize weights
                new_weights = dict(self._issue_weights)
                weights_sum = sum(self._issue_weights.values())
                for k, v in new_weights.items():
                    new_weights[k] = v / weights_sum
                self._issue_weights = new_weights

            # update current and previous window
            return DistributionBasedFrequencyOpponentModel(self._finished_first_window, self._domain,
                                                           self._issue_weights, self._current_window,
                                                           {iss: {} for iss in self._domain.getIssues()},
                                                           self._bidFrequencies, 0, self._resBid,
                                                           window_size=self._window_size, gamma=self._gamma,
                                                           alpha=self._alpha,
                                                           beta=self._beta)

        else:
            return DistributionBasedFrequencyOpponentModel(self._finished_first_window, self._domain,
                                                           self._issue_weights, self._prev_window,
                                                           self._current_window, self._bidFrequencies,
                                                           self._cw_bids_count,
                                                           self._resBid, self._window_size, gamma=self._gamma,
                                                           alpha=self._alpha, beta=self._beta)

    def _add_bid(self, bid: Bid):
        """
        Method updates the counts for current window and global frequencies with the given bid
        :param bid
        """
        self._cw_bids_count += 1

        # increase the counts for all the values in this bid
        current_window: Dict[str, Dict[Value, int]] = self.cloneMap(self._current_window)
        new_freqs: Dict[str, Dict[Value, int]] = self.cloneMap(self._bidFrequencies)
        for issue in self._domain.getIssues():
            value = bid.getValue(issue)

            cw_freqs: Dict[Value, int] = current_window[issue]  # current window issue freqs
            g_freqs: Dict[Value, int] = new_freqs[issue]  # global issue freqs
            if value is not None:
                # update count for current window
                cw_old_count = 0
                if value in cw_freqs:
                    cw_old_count = cw_freqs[value]
                cw_freqs[value] = cw_old_count + 1

                # update count for all offers (global)
                g_old_count = 0
                if value in g_freqs:
                    g_old_count = g_freqs[value]
                g_freqs[value] = g_old_count + 1
        return current_window, new_freqs

    @staticmethod
    def cloneMap(freqs: Dict[str, Dict[Value, int]]) -> Dict[str, Dict[Value, int]]:
        """
        @param freqs
        @return deep copy of freqs map.
        """
        map: Dict[str, Dict[Value, int]] = {}
        for issue in freqs:
            map[issue] = dict(freqs[issue])
        return map

    def _get_freq(self, freqs: Dict[Value, int], issue: str) -> List[float]:
        """
        Method returns a list of frequencies of all negotiation values of an issue for the passed window
        :param freqs: window of rounds for which the frequencies should be calculated
        :param issue
        """
        res: List[float] = list()

        num_values = self._domain.getValues(issue).size()

        for value in self._domain.getValues(issue):
            count = 0
            if value in freqs:
                count = freqs.get(value)
            res.append((1 + count) / (num_values + self._window_size))

        return res

    def _get_valuation(self, freqs: Dict[Value, int], issue: str, gamma: float = 0.25) -> List[float]:
        """
        Method returns a list of approximated valutions of all negotiation values of an issue for the passed window.
        :param freqs: window of rounds for which the frequencies should be calculated. Should be the global counts.
        :param issue
        :param gamma: 0 < gamma < 1, controls the growth of unbalanced value distributions when opponents send the same
        offer over and over for a significant part of the negotiation.
        """
        res: List[float] = list()

        # find max number of occurrences of an issue value
        max_count = get_issue_max_count(freqs)

        # calculate valuate for each value issue
        for value in self._domain.getValues(issue):
            count = get_value_val(value, freqs)
        return res

    def getValueVal(self, value: Value, issue):
        freqs = self._bidFrequencies[issue]
        return get_value_val(value,freqs)
    # Override
    def getReservationBid(self) -> Optional[Bid]:
        return self._resBid

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self._domain == other._domain and \
               self._issue_weights == other._issue_weights and \
               self._prev_window == other._prev_window and \
               self._current_window == other._current_window and \
               self._bidFrequencies == other._bidFrequencies and \
               self._cw_bids_count == other._cw_bids_count and \
               self._resBid == other._resBid and \
               self._window_size == other._window_size and \
               self._gamma == other._gamma and \
               self._alpha == other._alpha and \
               self._beta == other._beta

    def __hash__(self):
        return HASH((self._domain, self._bidFrequencies, self._totalBids, self._resBid))

    # Override

    # Override
    def __repr__(self) -> str:
        return "DistributionBasedFrequencyOpponentModel[" + str(self._totalBids) + "," + \
               toStr(self._bidFrequencies) + "]"
