import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_and_std_from_file(file, skiprows = None, drop_first = False):
    csv = pd.read_csv(file, header=None, skiprows = skiprows, names=list(range(39))).dropna(axis='columns', how='all')
    arr = np.array(csv)
    # if drop_first:
    #     arr = arr[:,1:]
    mean = np.nanmean(arr.astype(float),axis= 0)
    std = np.nanstd(arr.astype(float), axis = 0)
    return (mean,std)

def plotData(x, mean, std, label, plotStd = False,  errorevery = 1, alpha = 1.0, fmt= '-o'):
    if not plotStd :
        std = np.repeat(0,len(x))
    plt.errorbar(x, mean, yerr=std, errorevery = errorevery, fmt=fmt,label = label, alpha = alpha)


file_name = "time = 2022-06-14_095804, rounds = 1, opp = ['LinearAgent'].csv"
file_name = "time = 2022-06-14_150738, rounds = 1, opp = ['LinearAgent'].csv"
file_name = "time = 2022-06-14_151129, rounds = 1, opp = ['LinearAgent'].csv"
file_name = "time = 2022-06-14_152240, rounds = 1, opp = ['LinearAgent'].csv"
file_name = "time = 2022-06-14_152345, rounds = 10, opp = ['LinearAgent'].csv"
file_name = "time = 2022-06-14_152942, rounds = 10, opp = ['HardlinerAgent'].csv"
file_name = "time = 2022-06-14_203116, rounds = 1, opp = ['HardlinerAgent', 'BoulwareAgent', 'ConcederAgent', 'LinearAgent'].csv"
file_name = "time = 2022-06-14_204947, rounds = 1, opp = ['LinearAgent'].csv"
file_name = "time = 2022-06-14_210635, rounds = 1, opp = ['ConcederAgent'].csv"
file_name = "time = 2022-06-14_210726, rounds = 1, opp = ['HardlinerAgent'].csv"
(meanSmith , stdSmith) = mean_and_std_from_file(f"opp-model-logs/smith/{file_name}")
(meanPerceptron , stdPerceptron) = mean_and_std_from_file(f"opp-model-logs/perceptron/{file_name}")
(meanPPerceptron , stdPPerceptron) = mean_and_std_from_file(f"opp-model-logs/perfect_perceptron/{file_name}")
(meanReward, stdReward) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 1 ),drop_first= True)
(meanBidSpace, stdBidSpace) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 2 ),drop_first= True)
N = len(meanSmith)

x = np.arange(len(meanSmith))
plt.title("The evolution of the accuracy over the negotiation session")
plt.xlabel("Number of exchanged bids")
plt.ylabel("Pearson correlation of bids")
# mask = []
# mask.append (np.tile([0,0,1], (int((N+2)/3)))[:N])
# mask.append(np.tile([0,1,0], (int((N+2)/3)))[:N])
# mask.append(np.tile([1,0,0], (int((N+2)/3)))[:N])

n = 1000000
# plt.errorbar(x, meanSmith, yerr=stdSmith, fmt='-o',label = "Smith Model")
plotData(x[:n],meanSmith[:n],stdSmith[:n],"Smith Model", errorevery = (0,9) , plotStd = True)
plotData(x[:n],meanPerceptron[:n],stdPerceptron[:n],"Perceptron Model",  errorevery = (3,9), plotStd = True)
plotData(x[:n],meanPPerceptron[:n],stdPPerceptron[:n],"Perfect Perceptron Model", errorevery = (6,9), plotStd = True)
plotData(x[:n], meanReward[:n], stdReward[:n], "Opponent Reward" , plotStd=True, alpha=0.5,fmt = '-')
plotData(x[:n], 100*meanBidSpace[:n], stdBidSpace[:n], "Seen Bid Space" , plotStd=False, alpha=0.5,fmt = '-')
plt.legend()
plt.draw()
plt.savefig(f"graphs/{file_name[:-4]}")
plt.show()

