from os import path
from pickletools import pydict
from re import L
from typing import Any, Dict, Union, Set, Optional
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Value import Value
from geniusweb.utils import toStr


##initialises a 2d array of (numIssues x numValues)
## contains value/frequency pairs
##ASSUMING THAT THE CONTENTS OF THE DICTIONARIES ARE ALWAYS IN THE SAME ORDER
##THIS IS FULLY USELESS OTHERWISE
##IN THAT CASE IT SHOULD BE A DICTIONARY OF ((issue, value) -> frequency) PAIRS
def initProfileArr(agent):
    domain = agent._profile.getProfile().getDomain()
    issues = domain.getIssues()
    oppProfile = []
    for issue in issues:
        values = []
        for value in domain.getValues(issue):
            values.append((value, 0))
        oppProfile.append(values)
    return oppProfile


# creates a dictionary of ((issue, value) -> frequency) pairs
# this serves to count how many times the opponent picks each value
def initProfileDict(agent):
    domain = agent._profile.getProfile().getDomain()    #retrieves the domain
    issues = domain.getIssues()                         #retrieves the lsit of all issues
    oppProfile = {}
    for issue in issues:
        for value in domain.getValues(issue):           
            oppProfile[(issue, value)] = 0              #creates a kv pair for each value for each issue, initialised at 0
    return oppProfile

# takes a bid and updates the kv pair for each value chosen
def updateProfile(agent, bid: Bid):
    if bid == None: return
    for issue in bid.getIssues():                       #for each issue in the bid
        value = bid.getValue(issue)                     #retrieve the value chosen for that bid
        prev = agent.opponentProfile[(issue, value)]    #retrieve the previous frequency for that value
        agent.opponentProfile[(issue, value)] = prev+1  #update the dictionary with the frequency+1

def getIssueSanity(agent, issue, val):
    if(type(agent.opponentProfile) == type({})):
        domain = agent._profile.getProfile().getDomain()
        total = 0
        chosen = -1
        for value in domain.getValues(issue):
            total += agent.opponentProfile[(issue, value)]
            if value == val:
                chosen = agent.opponentProfile[(issue, value)]
        if chosen == -1:
            print("UH OH SOMETHING WENT REALLY WRONG :(")
            return -1.0
        else:
            return float(chosen)/float(total)
    else:
        #METHOD NOT WRITTEN FOR ARR
        return -1.0

def getSanity(agent, bid: Bid):
    total = 0
    for issue in bid.getIssues():
        value = bid.getValue(issue)
        total += getIssueSanity(agent, issue, value)
    return float(total) / len(bid.getIssues())

def compareBids(bid1: Bid, bid2: Bid):
    total = 0
    for issue in bid1.getIssues():
        if(bid1.getValue(issue) == bid2.getValue(issue)): 
            total += 1
    return float(total) / len(bid1.getIssues())

#
def makeMoveMats(agent):
    ret = []
    domain = agent._profile.getProfile().getDomain()
    issues = domain.getIssues()
    for issue in issues:
        rets = []
        for value in domain.getValues(issue):
            rets.append((agent.opponentProfile[(issue, value)], value))
        ret.append(sorted(rets, key=lambda x:x[0], reverse=True))
        
    return issues, ret

def oppProfileToString(agent):
    ret = "Opponent Profile:\n"
    domain = agent._profile.getProfile().getDomain()
    issues = domain.getIssues()
    for issue in issues:
        ret += issue + ":"
        for value in domain.getValues(issue):
                ret+= " (" + value.getValue()
                freq = agent.opponentProfile[(issue, value)]
                ret += ", " + str(freq) + ")"
        ret += "\n"
    return ret
    


def expandIndexes(indexes, digit):
    length = len(indexes)
    ret = indexes
    total = digit
    increments = []
    for i in range(len(indexes[0])):        #looping through possible ways of incrementing
        increments.append(0)
        val = 2^i
        if val < total:
            total -= val
            increments[i] = 1
        
    for i in range(length):
        # print(f"expandIndexes, i={i}, ret-{ret}")
        newIndexes = []
        for j in range(len(indexes[i])):
            # print(indexes[i][j])
            newIndexes.append(indexes[i][j]+increments[j])
        ret.append(newIndexes)
    # print(f"returning {ret}")
    return ret

def loadFile(agent):
    if path.exists(agent.tempDir):
        file = open(agent.tempDir, "r")
        ret = file.read()
        file.close()
        return ret
    else: return ""


def parseFile(agent):
    ret = []
    for line in agent.file.split("\n"):
        if len(line) <= 2: continue
        val = []
        for x in line.split(" "):
            val.append(float(x))
        ret.append(val)
    return ret


def recalculateSBS(agent, lines):
    length = len(lines)
    if length==0: return 0
    newSBS = 0.0
    for line in lines:
        if line[0] >= 0.99:
            newSBS += 0.65
        elif line[0] >= line[1]:
            newSBS += line[1]
        else:
            newSBS += 0.95
    newSBS = newSBS / length
    return newSBS


def getLine(agent):
    progress = agent._progress.get(0)
    SBS = agent.smartBidStart
    line = f"\n{progress} {SBS}"
    return agent.file + line

def writeFile(agent):
    file = open(agent.tempDir, "w")
    file.write(getLine(agent))
    file.close()