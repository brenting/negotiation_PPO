import numpy as np
from geniusweb.issuevalue.Bid import Bid

class OpponentModel:

    def __init__(self, profile):
        self._roundNumber = 0
        self._biddingHistory = []
        self._utilityBiddingHistory = None
        self._updatesTotal = 1
        self._concessionRatioDistributions = []
        self._reservationValue = 0
        self._windows = []
        self._profile = profile
        self._issues = profile.getProfile().getDomain().getIssues()
        self._issueWeights = np.ones(len(self._issues))/len(self._issues)
        self._window_size = 20
        self.frequencyModel = None
        self._frequencyModelWeightPredictions = []


    def getIssueWeights(self):
        res = dict()
        counter = 0
        for issue in self._issues:
            res[issue] = self._issueWeights[counter]
            counter += 1
        return res

    def updateBiddingHistory(self, newBid: Bid):
        self._roundNumber+=1
        self._biddingHistory.append(newBid)
        if len(self._biddingHistory) > 1:
            self.updateIssueWeights()
        self._frequencyModelWeightPredictions.append(list(self.frequencyModel.getIssueWeights().values()))

    def getOpponentUtility(self, potentialBid: Bid):
        sum = 0
        issueValues = potentialBid.getIssueValues()
        issueWeights = self._issueWeights #

        for counter, issue in enumerate(self._issues):
            sum += issueWeights[counter] * self.frequencyModel.getValueVal(issueValues[issue], issue)
        return sum

    def updateIssueWeights(self):
        # first step: we compare 2 consecutive bids, and measure how much each issue decreases in terms of proportion.

        estimatedIssueWeights = np.empty(len(self._issues))


        last20concessionRatios = self.getLast20ConcessionRatios()
        if(last20concessionRatios == None):
            return self._issueWeights

        concessionRatios = last20concessionRatios[-1]
        for counter, concessionRatio in enumerate(concessionRatios):
            estimatedIssueWeights[counter] = self.mapConcessionRatiosToIssueWeights(last20concessionRatios,concessionRatio)

        estimatedIssueWeights = estimatedIssueWeights / np.sum(estimatedIssueWeights)

        frequencyModelWeights = np.array(list(self.frequencyModel.getIssueWeights().values()))
        self._issueWeights = self._issueWeights + (estimatedIssueWeights - self._issueWeights) / self._updatesTotal
        self._issueWeights = self._issueWeights * 0.2 + frequencyModelWeights * 0.8
        self._issueWeights = self._issueWeights / np.sum(self._issueWeights)
        if(self._updatesTotal <10):
            self._updatesTotal += 1

    def getLast20ConcessionRatios(self):
        currentValues = self._biddingHistory[max(0, self._roundNumber - 20)].getIssueValues()
        last20concessionRatios = []
        concessionRatios = np.empty(len(self._issues))
        for x in range(max(1, self._roundNumber - 19), self._roundNumber):
            previousValues = currentValues
            currentValues = self._biddingHistory[x].getIssueValues()

            for counter, issue in enumerate(self._issues):

                ratio = self.frequencyModel.getValueVal(currentValues[issue], issue) / self.frequencyModel.getValueVal(previousValues[issue], issue)

                concessionRatios[counter] = ratio
            last20concessionRatios.append(concessionRatios)
        return last20concessionRatios

    def mapConcessionRatiosToIssueWeights(self, last20ConcessionRatios, concessionRatio):

        vectorizedConcessionRatioDistributions = np.array(last20ConcessionRatios).flatten()


        percentile = int(len(
            vectorizedConcessionRatioDistributions[vectorizedConcessionRatioDistributions < concessionRatio])
                         * 99 / len(vectorizedConcessionRatioDistributions))
        return np.percentile(self._frequencyModelWeightPredictions, 100 - percentile)



    def estimateReservationValue(self):
        self.updateUtilities()
        self.detectWindows()

        # the convolution reduce some noise
        # todo maybe do some fun stuff with fourier, but gotta see first
        utilityBiddingHistory = np.convolve(self._utilityBiddingHistory, np.ones(5) / 5)

        windowStart = 0

        reservationValueEstimates = []

        for windowFinish in self._windows:
            maxLag = windowStart - windowFinish

            # every possible lag between 2 bids
            for lag in range(maxLag):
                tuples = np.empty(windowFinish - lag - windowStart)

                # storing this in tuples of 2 bids
                for x in range(windowStart, windowFinish - lag):
                    tuples[x - windowStart] = (utilityBiddingHistory[x], utilityBiddingHistory[x + lag])
                for x in range(len(tuples) - 2):
                    for y in range(x + 1, len(tuples)):
                        (u1_x, u2_x) = tuples[x]
                        (u1_y, u2_y) = tuples[y]
                        reservationValueEstimate = (u1_x * u2_y - u2_x * u1_y) / (u1_x - u2_x + u1_y - u2_y)
                        # only add reservation value if it makes sense (so it's not negative or above the latest utility they offered)
                        if (reservationValueEstimate <= utilityBiddingHistory[-1] and reservationValueEstimate >= 0):
                            reservationValueEstimates.append(reservationValueEstimate)
        reservationValueEstimates = np.array(reservationValueEstimate)
        return np.mean(reservationValueEstimates)

    def detectWindows(self):
        firstDerivative = self._utilityBiddingHistory[0:-2] - self._utilityBiddingHistory[1:-1]
        secondDerivative = np.abs(firstDerivative[0:-2] - firstDerivative[1:-1])
        top20Percent = np.percentile(secondDerivative, 80)

        windows = np.sort(np.argwhere(secondDerivative > top20Percent))

        previousWindowStart = 0

        # returns only windows that are at least 10 rounds apart
        for x in range(len(windows)):
            if (previousWindowStart + 10 > windows[x]):
                windows[x] = -1
            else:
                previousWindowStart = windows[x]

        return windows[windows != -1]

    def updateUtilities(self):
        utilities = np.empty(len(self._biddingHistory))
        for x, bid in enumerate(self._biddingHistory):
            utilities[x] = self.getOpponentUtility(bid)

        self._utilityBiddingHistory = utilities
