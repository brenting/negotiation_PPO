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
    mean = np.nanmean(arr.astype(float))
    # max = np.max(np.nanmean(arr.astype(float),axis = 0))
    max = np.mean(np.nanmax(arr.astype(float),axis = 1))
    return (mean,max)

#taken from https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []
    
    #Create second Y-axis
    ax2 = ax.twinx()


    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            if name == "Seen Bid Space":
                bar = ax2.bar(x + x_offset, y, width=bar_width * single_width, color=colors[3])
            else:
                bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[0])
                ax.text(x + x_offset-0.08, 0.005, name, rotation = -90, fontsize = 11, va = 'bottom', color='seashell',)

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    ax.set_ylabel('Accuracy', c = colors[0])
    ax.set_ylim([0, 1])
    ax2.set_ylabel('% Bid Space', c = colors[3])
    # Draw legend if we need
    if legend:
        ax.legend(bars[::len(bars)-1], ["Accuracy of the models","Seen bid Space"],loc='upper left')



data1 = {
    "Smith" : [],
    "Bad Perceptron" : [],
    "Perfect Perceptron" : [],
    "Seen Bid Space" : []
 }

data2 = {
    "Smith" : [],
    "Bad Perceptron" : [],
    "Perfect Perceptron" : [],
    "Seen Bid Space" : []
 }
file_names = []
file_name = "time = 2022-06-15_064453, rounds = 500, opp = ['HardlinerAgent'].csv"
file_names.append(file_name)
file_name = "time = 2022-06-15_145603, rounds = 500, opp = ['ConcederAgent'].csv"
file_names.append(file_name)
file_name = "time = 2022-06-14_232443, rounds = 500, opp = ['BoulwareAgent'].csv"
file_names.append(file_name)
file_name = "time = 2022-06-14_211332, rounds = 500, opp = ['LinearAgent'].csv"
file_names.append(file_name)

for file_name in file_names:
    (meanSmith , maxSmith) = mean_and_std_from_file(f"opp-model-logs/smith/{file_name}")
    (meanPerceptron , maxPerceptron) = mean_and_std_from_file(f"opp-model-logs/perceptron/{file_name}")
    (meanPPerceptron , maxPPerceptron) = mean_and_std_from_file(f"opp-model-logs/perfect_perceptron/{file_name}")
    (meanBidSpace, maxBidSpace) = mean_and_std_from_file(f"opp-model-logs/round_info/{file_name}",skiprows= (lambda x : x % 3 != 2 ),drop_first= True, c = 100)
    data1["Smith"].append(meanSmith)
    data1["Bad Perceptron"].append(meanPerceptron)
    data1["Perfect Perceptron"].append(meanPPerceptron)
    data1["Seen Bid Space"].append(meanBidSpace)

    data2["Smith"].append(maxSmith)
    data2["Bad Perceptron"].append(maxPerceptron)
    data2["Perfect Perceptron"].append(maxPPerceptron)
    data2["Seen Bid Space"].append(maxBidSpace)




opponents = ["Hardliner", "Conceder", "Boulware", "Linear"]
fig, ax = plt.subplots()
bar_plot(ax, data1, total_width=.8, single_width=.9)
print(data1)
plt.xticks(range((len(opponents))), opponents)
plt.title("The average accuracy and explored bid space")
plt.savefig("graphs/Bars1.png",dpi=300)
plt.show()
# fig, ax = plt.subplots()
# bar_plot(ax, data2, total_width=.8, single_width=.9)
# plt.xticks(range((len(opponents))), opponents)
# plt.title("The maximum accuracy and explored bid space")
# plt.savefig("graphs/Bars2.png",dpi=300)
# plt.show()

bids = data1["Seen Bid Space"]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, (name, values) in enumerate(data1.items()):
    if(i < 3):
        bids = data1["Seen Bid Space"]
        x,y = zip(*sorted(zip(bids, values)))
        plt.plot(x,y,linestyle="-", marker="o", c=colors[i],label = name,alpha = 0.9)
plt.ylabel("Accuracy")
plt.xlabel("% Bid Space")
plt.legend(loc='upper left')
plt.title("Correlation between accuracy and explored bid space")
# xposition = data1["Seen Bid Space"]
# for xc,c in zip(xposition,colors[1:]):
#     plt.axvline(x=xc, c="gray" ,linestyle='--')
# # plt.errorbar(x, meanSm


plt.savefig("graphs/Scatter.png",dpi=300)




plt.show()