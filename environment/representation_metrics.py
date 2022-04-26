import json
import numpy as np
from environment.domains import get_domains


def distance_issue_weights(p_actual, p_modeled):
    f = open(p_actual + "/profileA.json", "r")
    f2 = open(p_modeled, "r")

    actual = json.load(f)["LinearAdditiveUtilitySpace"]["issueWeights"]
    modeled = json.load(f2)["LinearAdditiveUtilitySpace"]["issueWeights"]

    weights_actual = np.array(list(actual.values()))
    weights_modeled = np.array(list(modeled.values()))
    res = np.sqrt(np.sum((weights_actual - weights_modeled) ** 2))
    return res


def distance_issue_values(p_actual, p_modeled):
    f = open(p_actual + "/profileA.json", "r")
    f2 = open(p_modeled, "r")

    actual = json.load(f)
    modeled = json.load(f2)

    issues = modeled["LinearAdditiveUtilitySpace"]["issueUtilities"]
    actual_issues = actual["LinearAdditiveUtilitySpace"]["issueUtilities"]
    total = 0
    for issue in issues.keys():
        values = issues[issue]["DiscreteValueSetUtilities"]["valueUtilities"]
        actual_values = actual_issues[issue]["DiscreteValueSetUtilities"]["valueUtilities"]
        utils = np.array(list(values.values()))
        actual_utils = np.array(list(actual_values.values()))
        total += np.sqrt(np.sum((actual_utils - utils) ** 2))

    total /= len(issues)
    return total

