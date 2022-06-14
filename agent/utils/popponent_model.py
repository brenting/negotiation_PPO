from errno import EISDIR
from matplotlib.pyplot import get
import numpy as np
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Bid import Bid



class PerceptronOpponentModel:
    def __init__(self, domain: Domain):
        self.N = 50
        self.learning_rate = 0.8
        self.offers = []
        self.domain = domain
        self.size = len(domain.getIssuesValues())
        self.weights = {
            issueName: 1.0/self.size for issueName,issueValueSet in domain.getIssuesValues().items()
        }
        self.issueValues = {
            issueName: {key : 0 for key in issueValueSet.getValues()} for issueName,issueValueSet in domain.getIssuesValues().items()
        }
        # print("Initializing Perceptron opponent model for " +str(self.N) + " issues")

    def update(self, bid: Bid, estimatedUtility):
        self.offers.append(bid)
        if estimatedUtility == None:
            return
        for epoch in range(self.N):
            for issueName,issueValue in bid.getIssueValues().items():
                perceptronUtility = self.get_predicted_utility(bid)
                if(self.issueValues[issueName][issueValue] == 0):
                    self.issueValues[issueName][issueValue] = 0.1
                self.issueValues[issueName][issueValue] = self.issueValues[issueName][issueValue] + self.learning_rate * (estimatedUtility - perceptronUtility) * self.issueValues[issueName][issueValue]
            for issueName in self.domain.getIssues():
                perceptronUtility = self.get_predicted_utility(bid)
                self.weights[issueName] = self.weights[issueName] + self.learning_rate * (estimatedUtility - perceptronUtility) * self.weights[issueName]
            
    def get_history(self):
        return self.offers
    def get_predicted_utility(self, bid: Bid):

        value_utilities = []
        issue_weights = []
        for issueName, issueValue in bid.getIssueValues().items():
            value_utilities.append(self.issueValues[issueName][issueValue])
            issue_weights.append(self.weights[issueName])
            

        # calculate predicted utility by multiplying all value utilities with their issue weight
        predicted_utility = sum(
            [iw * vu for iw, vu in zip(issue_weights, value_utilities)]
        )
        return predicted_utility