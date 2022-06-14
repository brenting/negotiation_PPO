from math import sqrt
from tabnanny import verbose
import timeit
from geniusweb.bidspace import AllBidsList
from geniusweb.issuevalue.Domain import Domain
import numpy as np
class PearsonCorrelation:

    def __init__(self, domain: Domain, real_utility_fun):
        self.domain = domain
        self.allBids = list(AllBidsList.AllBidsList(domain))
        self.verbose = False
        self.real_utility_fun = real_utility_fun
        self.real_utility = np.array(list(map(lambda x : (float)(self.real_utility_fun(x)), self.allBids)))

    def getBidSpaceSize(self):
        return len(self.allBids)

    def pearsonCorrelationOfBids(self, predicted_utility_fun):
        start = timeit.default_timer()
        real_utility = self.real_utility
        predicted_utility =np.array(list(map(lambda x : (float)(predicted_utility_fun(x)), self.allBids)))
        # print(type(np.array(list(real_utility))))
        average_real_utility = np.average(real_utility)
        average_predicted_utility = np.average(predicted_utility)
        #this is the top component of the pearson coefficient
        real_norm = real_utility - average_real_utility
        pred_norm = predicted_utility - average_predicted_utility
        sum_of_products = np.sum(np.multiply(real_norm, pred_norm)) 
        #this are the bottom components of the pearson coefficient
        realVar = np.sum(np.multiply(real_utility - average_real_utility, real_utility - average_real_utility))
        predictedVar = np.sum(np.multiply(predicted_utility - average_predicted_utility, predicted_utility - average_predicted_utility))
        if(realVar != 0 and predictedVar != 0):
            pearsonCorrelation = sum_of_products / sqrt(realVar * predictedVar)
        else:
            pearsonCorrelation = 0
        stop = timeit.default_timer()

        if self.verbose:
            diff = np.abs(real_utility - predicted_utility)
            print("Difference " , diff)
            print("Absolute ", np.sum(diff))
            print("Real ", average_real_utility,real_utility)
            print("Estimated ", average_predicted_utility, predicted_utility)
            print("Real norm", real_norm)
            print("Pred norm", pred_norm)
            print("\033[1;32;40mTime to calculate pearson correlation: ", stop - start , "result: ", pearsonCorrelation)  

        return pearsonCorrelation


    def pearsonCorrelationOfBidsSlowVersion(self, predicted_utility_fun):
        #print(actual_utility.getDomain)
        #print(predicted_utility.domain)
        start = timeit.default_timer()
        bids = AllBidsList.AllBidsList(self.domain)
        #print("Number of bids in bid space: " + str(bids.size()))
        sum_real_utility = 0.0
        sum_predicted_utility = 0.0
        for i in range(bids.size()):
            sum_real_utility += float(self.real_utility_fun(bids.get(i)))
            sum_predicted_utility += predicted_utility_fun(bids.get(i))
        average_real_utility = sum_real_utility / bids.size()
        average_predicted_utility = sum_predicted_utility / bids.size()

        #this is the top component of the pearson coefficient
        sum_of_products = 0.0
        #this are the bottom components of the pearson coefficient
        realVar = 0.0
        predictedVar = 0.0
        for i in range(bids.size()):
            sum_of_products += (float(self.real_utility_fun(bids.get(i)))-average_real_utility)*(predicted_utility_fun(bids.get(i))-average_predicted_utility)
            realVar += (float(self.real_utility_fun(bids.get(i)))-average_real_utility)**2
            predictedVar += (predicted_utility_fun(bids.get(i))-average_predicted_utility)**2
        if(realVar != 0 and predictedVar != 0):
            pearsonCorrelation = sum_of_products / sqrt(realVar * predictedVar)
        else:
            pearsonCorrelation = 0
        stop = timeit.default_timer()

        if verbose:
            print("\033[1;34;40m Time to calculate pearson correlation: ", stop - start , "result: ", pearsonCorrelation)  

        return pearsonCorrelation