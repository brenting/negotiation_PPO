import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_and_std_from_file(file):
    csv = pd.read_csv(file, header=None, names=list(range(300))).dropna(axis='columns', how='all')
    arr = np.array(csv)
    mean = np.nanmean(arr,axis= 0)
    std = np.nanstd(arr, axis = 0)
    return (mean,std)


# file_name = ", N=2 rounds, opp = ['BoulwareAgent'], time = 2022-06-02_202910.csv"
# file_name = ", N=10 rounds, opp = ['BoulwareAgent'], time = 2022-06-02_201407.csv"
#file_name = ", N=200 rounds, opp = ['BoulwareAgent'], time = 2022-06-02_212038.csv"
# file_name = ", N=200 rounds, opp = ['HardlinerAgent'], time = 2022-06-02_223858.csv"
#file_name = ", N=500 rounds, opp = ['HardlinerAgent'], time = 2022-06-03_080412.csv"
# file_name = ", N=100 rounds, opp = ['BoulwareAgent', 'ConcederAgent', 'HardlinerAgent', 'LinearAgent', 'RandomAgent', 'StupidAgent'], time = 2022-06-08_083629.csv"
# file_name = ", N=100 rounds, opp = ['BoulwareAgent', 'ConcederAgent', 'HardlinerAgent', 'LinearAgent'], time = 2022-06-08_091140.csv"
# file_name = ", N=50 rounds, opp = ['HardlinerAgent'], time = 2022-06-08_094528.csv"
# # file_name = ", N=150 rounds, opp = ['HardlinerAgent', 'BoulwareAgent'], time = 2022-06-08_095050.csv"
# # file_name = ", N=100 rounds, opp = ['BoulwareAgent'], time = 2022-06-08_101151.csv"
# #file_name = ", N=300 rounds, opp = ['BoulwareAgent', 'HardlinerAgent', 'LinearAgent', 'ConcederAgent'], time = 2022-06-08_103118.csv"
# file_name = ", N=500 rounds, opp = ['BoulwareAgent', 'HardlinerAgent', 'LinearAgent', 'ConcederAgent'], time = 2022-06-08_111624.csv"

file_name = "time = 2022-06-09_130324, rounds = 50, opp = ['BoulwareAgent', 'HardlinerAgent', 'LinearAgent', 'ConcederAgent'].csv"
(meanSmith , stdSmith) = mean_and_std_from_file(f"opp-model-logs/smith/{file_name}")
(meanPerceptron , stdPerceptron) = mean_and_std_from_file(f"opp-model-logs/perceptron/{file_name}")
(meanPPerceptron , stdPPerceptron) = mean_and_std_from_file(f"opp-model-logs/perfect_perceptron/{file_name}")

N = len(meanSmith)

x = np.arange(len(meanSmith))
plt.title("The evolution of the accuracy over the negotiation session")
plt.xlabel("Number of exchanged bids")
plt.ylabel("Pearson correlation of bids")
mask = np.tile([0,0,1], (int((N+2)/3)))[:N]
# plt.errorbar(x, meanSmith, yerr=stdSmith, fmt='-o',label = "Smith Model")
plt.errorbar(x, meanSmith, yerr=(stdSmith*mask), fmt='-o',label = "Smith Model")
mask = np.tile([0,1,0], (int((N+2)/3)))[:N]
# plt.errorbar(x, meanPerceptron,yerr = stdPerceptron, fmt='-o',label = "Perceptron Model")
plt.errorbar(x, meanPerceptron,yerr = (stdPerceptron * mask), fmt='-o',label = "Perceptron Model")

mask = np.tile([1,0,0], (int((N+2)/3)))[:N]
plt.errorbar(x, meanPPerceptron,yerr = (stdPPerceptron * mask), fmt='-o',label = "Perfect Perceptron Model")
plt.legend()
plt.draw()
plt.show()

