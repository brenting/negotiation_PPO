from turtle import xcor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_and_std_from_file(file, skiprows = None, drop_first = False, c = None):
    csv = pd.read_csv(file, header=None, skiprows = skiprows, names=list(range(300))).dropna(axis='columns', how='all')
    arr = np.array(csv)
    if drop_first:
         arr = arr[:,1:]
    if c != None:
        arr *= c
    aux = np.array(arr)
    aux[pd.isna(aux)] = 0
    aux[aux > 0] = 1
    aux = np.sum(aux, axis= 0)
    mean = np.nanmean(arr.astype(float),axis= 0)
    std = np.nanstd(arr.astype(float), axis = 0)
    return (mean,std, aux)


def plotData(x, mean, std, label, plotStd = False,  errorevery = 1, alpha = 1.0, fmt= '-o'):
    if not plotStd :
        std = np.repeat(0,len(x))
    plt.errorbar(x, mean, yerr=std, errorevery = errorevery, fmt=fmt,label = label, alpha = alpha)

n = 1000000
plotStd = False

file_name = "time = 2022-06-15_064453, rounds = 500, opp = ['HardlinerAgent'].csv"
(meanBidSpace, stdBidSpace,aux) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 2 ),drop_first= True, c = 100)
print("Hardliner ", str(list(zip(range(10000),aux))))
x = np.arange(len(meanBidSpace))
plotData(x[:n], meanBidSpace[:n], stdBidSpace[:n], "Hardliner Agent" , plotStd=plotStd, alpha=0.9,fmt = '-',errorevery = (2,4))

file_name = "time = 2022-06-15_145603, rounds = 500, opp = ['ConcederAgent'].csv"
(meanBidSpace, stdBidSpace,aux) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 2 ),drop_first= True, c = 100)
print("Conceder", str(list(zip(range(10000),aux))))
x = np.arange(len(meanBidSpace))
plotData(x[:n], meanBidSpace[:n], stdBidSpace[:n], "Conceder Agent" , plotStd=plotStd, alpha=0.9,fmt = '-',errorevery = (3,4))

file_name = "time = 2022-06-14_232443, rounds = 500, opp = ['BoulwareAgent'].csv"
(meanBidSpace, stdBidSpace,aux) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 2 ),drop_first= True, c = 100)
print("Boulware ", str(list(zip(range(10000),aux))))
x = np.arange(len(meanBidSpace))
plotData(x[:n], meanBidSpace[:n], stdBidSpace[:n], "Boulware Agent" , plotStd=plotStd, alpha=0.9,fmt = '-',errorevery = (1,4))

file_name = "time = 2022-06-14_211332, rounds = 500, opp = ['LinearAgent'].csv"
(meanBidSpace, stdBidSpace,aux) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 2 ),drop_first= True, c = 100)
print("Linear ", str(list(zip(range(10000),aux))))
x = np.arange(len(meanBidSpace))
plotData(x[:n], meanBidSpace[:n], stdBidSpace[:n], "Linear Agent" , plotStd=plotStd, alpha=0.9,fmt = '-',errorevery = (0,4))

plt.title("The evolution of the explored bid space for all oppponents")    
plt.ylabel("% bid space")
plt.xlabel("Number of exchanged bids")
#plt.ylabel("Pearson correlation of bids")
# mask = []
# mask.append (np.tile([0,0,1], (int((N+2)/3)))[:N])
# mask.append(np.tile([0,1,0], (int((N+2)/3)))[:N])
# mask.append(np.tile([1,0,0], (int((N+2)/3)))[:N])

xposition = [88,149,123]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for xc,c in zip(xposition,colors[1:]):
    plt.axvline(x=xc, c=c ,linestyle='--')
# plt.errorbar(x, meanSmith, yerr=stdSmith, fmt='-o',label = "Smith Model")

plt.legend()
plt.draw()
plt.savefig("graphs/Bid Space.png",dpi=300)
plt.show()

