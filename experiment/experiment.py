from cgitb import reset
import copy
import time
from pathlib import Path
from experiment.pearson_correlation import PearsonCorrelation
from environment.negotiation import NegotiationEnv
class OpponentModellingExperiment:

    def __init__(self, opponents,N) -> None:
        
        self.estimatedUtility = None
        
        log_dir_path = []

        #path to the directories
        log_dir_path.append(Path("opp-model-logs/smith"))
        log_dir_path.append(Path("opp-model-logs/perceptron"))
        log_dir_path.append(Path("opp-model-logs/perfect_perceptron"))

        # create results directory if it does not exist
        for dir in log_dir_path:
            if not dir.exists():
                dir.mkdir(parents=True)

        # track total training time
        start_time = time.strftime("%Y-%m-%d_%H%M%S")
        opp_names = str(list(map(lambda x : str(x).split(".")[-1][:-2], opponents)))
        file_name = f"time = {start_time}, rounds = {N}, opp = {opp_names}.csv"
        self.logs = []
        for dir in log_dir_path:
            temp = open(dir.joinpath(file_name), "w+")
            self.logs.append(temp)

    def saveModels(self,agent) -> None:
        if self.estimatedUtility == None:
            self.estimatedUtility = [[],[],[]]
        #making a deep copy of the estimated opponent model(s)
        self.estimatedUtility[0].append(copy.deepcopy(agent.opponent_model))
        self.estimatedUtility[1].append(copy.deepcopy(agent.opponent_model2))
        self.estimatedUtility[2].append(copy.deepcopy(agent.opponent_model3))

    def saveResults(self,env: NegotiationEnv) -> None:
        self.corr = PearsonCorrelation(env.opp_utility_function.getDomain(),env.opp_utility_function.getUtility)
        for index,estimatedUtility in enumerate(self.estimatedUtility):
            result = self.calculateResults(estimatedUtility)
            arrayWithoutBrackets = str(result).replace(" [","").replace("[","").replace("]","")
            self.logs[index].write(f"{arrayWithoutBrackets}\n")
            self.logs[index].flush()
        print("Writing logs")
        self.estimatedUtility = None
        

    def calculateResults(self, estimatedUtility):
        result = []
        for index, f in enumerate(estimatedUtility):  
            result.append(self.corr.pearsonCorrelationOfBids(f.get_predicted_utility))
        return result

        