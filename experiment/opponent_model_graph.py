import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_and_std_from_file(file, skiprows = None, drop_first = False):
    csv = pd.read_csv(file, header=None, skiprows = skiprows, names=list(range(300))).dropna(axis='columns', how='all')
    arr = np.array(csv)
    if drop_first:
         arr = arr[:,1:]
    mean = np.nanmean(arr.astype(float),axis= 0)
    std = np.nanstd(arr.astype(float), axis = 0)
    return (mean,std)

def plotData(x, mean, std, label, plotStd = False,  errorevery = 1, alpha = 1.0, fmt= '-o'):
    if not plotStd :
        std = np.repeat(0,len(x))
    plt.errorbar(x, mean, yerr=std, errorevery = errorevery, fmt=fmt,label = label, alpha = alpha)

file_names = []
file_names.append("time = 2022-06-15_064453, rounds = 500, opp = ['HardlinerAgent'].csv")
file_names.append("time = 2022-06-15_145603, rounds = 500, opp = ['ConcederAgent'].csv")
file_names.append("time = 2022-06-14_232443, rounds = 500, opp = ['BoulwareAgent'].csv")
file_names.append("time = 2022-06-14_211332, rounds = 500, opp = ['LinearAgent'].csv")
file_names.append("time = 2022-06-16_130812, rounds = 500, opp = ['HardlinerAgent', 'BoulwareAgent', 'ConcederAgent', 'LinearAgent'].csv")


meanRewards = []
stdRewards = []

names = ["Hardliner","Conceder","Boulware","Linear"]
for ind, file_name in enumerate(file_names):
    (meanSmith , stdSmith) = mean_and_std_from_file(f"opp-model-logs/smith/{file_name}")
    (meanPerceptron , stdPerceptron) = mean_and_std_from_file(f"opp-model-logs/perceptron/{file_name}")
    (meanPPerceptron , stdPPerceptron) = mean_and_std_from_file(f"opp-model-logs/perfect_perceptron/{file_name}")
    (meanReward, stdReward) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 1 ),drop_first= True)
    (meanBidSpace, stdBidSpace) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 2 ),drop_first= True)
    N = len(meanSmith)
    meanRewards.append(meanReward)
    stdRewards.append(stdReward)
    x = np.arange(len(meanSmith))
    if(ind < 4):
        plt.title(f"The evolution of the accuracy against the {names[ind]} agent")
    else:
        plt.title(f"The evolution of the accuracy against all opponents")    
    plt.xlabel("Number of exchanged bids")
    plt.ylabel("Accuracy")
    # mask = []
    # mask.append (np.tile([0,0,1], (int((N+2)/3)))[:N])
    # mask.append(np.tile([0,1,0], (int((N+2)/3)))[:N])
    # mask.append(np.tile([1,0,0], (int((N+2)/3)))[:N])

    n = 1000000
    # plt.errorbar(x, meanSmith, yerr=stdSmith, fmt='-o',label = "Smith Model")
    plotData(x[:n],meanSmith[:n],stdSmith[:n],"Smith Model", errorevery = (0,9) , plotStd = True)
    plotData(x[:n],meanPerceptron[:n],stdPerceptron[:n],"Bad Perceptron Model",  errorevery = (3,9), plotStd = True)
    plotData(x[:n],meanPPerceptron[:n],stdPPerceptron[:n],"Perfect Perceptron Model", errorevery = (6,9), plotStd = True)
    # plotData(x[:n], meanReward[:n], stdReward[:n], "Opponent Reward" , plotStd=True, alpha=0.5,fmt = '-')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    # plotData(x[:n], meanBidSpace[:n], stdBidSpace[:n], "Seen Bid Space" , plotStd=False, alpha=0.5,fmt = '-')
    plt.legend(loc='lower left')
    plt.draw()
    plt.savefig(f"graphs/{file_name[:-4]}", dpi = 300)
    plt.show(block=False)
    plt.close()

names = ["Hardliner","Conceder","Boulware","Linear"]
for i , (meanReward,stdReward) in enumerate(zip(meanRewards[:-1],stdRewards[:-1])):
    plotData(np.arange(len(meanReward))[:n], meanReward[:n], stdReward[:n], names[i] , alpha=1, plotStd=False, errorevery = (i,4))


plt.title("The evolution of the utility goal for all oppponents")    
plt.xlabel("Number of exchanged bids")
plt.ylabel("Utility")
plt.legend(loc='lower left')
plt.draw()
plt.savefig(f"graphs/Utility", dpi = 300)
plt.show()