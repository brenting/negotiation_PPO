import json
import math
import os
import numpy as np
from environment.representation_metrics import distance_issue_weights, distance_issue_values

# Use either test/train
folder = "test"


def fuzz_issue_weights(folder):
    domains = [f.path for f in os.scandir("./domains/" + folder) if f.is_dir()]
    d_number = 0
    for d in domains:

        f = open(d + "/profileA.json", "r")
        profile = json.load(f)
        weights = np.array(list(profile["LinearAdditiveUtilitySpace"]["issueWeights"].values()))
        # add random values
        updated = weights + np.random.random_sample((len(weights),))
        # normalize so sum of weights is 1
        updated /= (np.sum(updated))
        assert (math.isclose(np.sum(updated), 1.0))

        # update dict
        count = 0
        for issue in profile["LinearAdditiveUtilitySpace"]["issueWeights"]:
            profile["LinearAdditiveUtilitySpace"]["issueWeights"][issue] = updated[count]
            count += 1

        # write to file
        out_file = open("models/fuzzed_weights/" + folder + "/" + str(d_number) + ".json", "w")
        json.dump(profile, out_file)
        out_file.close()

        # compute distance metric
        #print(distance_issue_weights(d, "models/fuzzed_weights/" + folder + "/" + str(d_number) + ".json"))
        d_number += 1


def fuzz_value_utilities(folder):
    domains = [f.path for f in os.scandir("./domains/" + folder) if f.is_dir()]
    d_number = 0
    for d in domains:

        f = open(d + "/profileA.json", "r")
        profile = json.load(f)
        issues = profile["LinearAdditiveUtilitySpace"]["issueUtilities"]
        for issue in issues.keys():
            values = issues[issue]["DiscreteValueSetUtilities"]["valueUtilities"]
            utils = np.array(list(values.values()))
            updated = utils + np.random.random_sample((len(utils),))
            updated /= np.max(updated)
            assert (math.isclose(np.max(updated), 1.0))
            count = 0
            for value in values:
                profile["LinearAdditiveUtilitySpace"]["issueUtilities"][issue]["DiscreteValueSetUtilities"][
                    "valueUtilities"][value] = updated[count]
                count += 1

        # write to file
        out_file = open("models/fuzzed_values/" + folder + "/" + str(d_number) + ".json", "w")
        json.dump(profile, out_file)
        out_file.close()
        #print(distance_issue_values(d, "models/fuzzed_values/" + folder + "/" + str(d_number) + ".json"))
        d_number += 1


fuzz_issue_weights(folder)
fuzz_value_utilities(folder)
