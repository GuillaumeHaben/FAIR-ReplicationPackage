import os
import sys
import time
import json
import pandas as pd
import seaborn as sn
from pprint import pprint
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    """
    Main function
    """
    checkUsage()
    with open(sys.argv[1]) as testsInfo:
        dataDic = json.load(testsInfo)
    
    print("--- Chromium Analysis | infoRuns ---\n")

    data = pd.read_json(sys.argv[1])

    flaky = data[data["label"] == 0]
    failures = data[data["label"] == 1]

    reliableFailures = failures[(failures["flakeRate"] == 0)] 
    unreliableFailures = failures[(failures["flakeRate"] > 0)]
    
    # Table 2
    print("Flaky quantile:", flaky.runDuration.quantile([0.25, 0.5, 0.75]))
    print("Failures quantile:",failures.runDuration.quantile([0.25, 0.5, 0.75]))

    # General info
    print("Number of flaky runs", len(flaky))
    print("Number of failure runs", len(failures))
    print("Number of reliable failures", len(reliableFailures))
    print("Number of unreliable failures", len(unreliableFailures))

    # Figure 1
    rates(data)

def rates(data):
    # Rename label
    data['label'] = data['label'].replace([0],'False alerts')
    data['label'] = data['label'].replace([1],'Legitimate failures')
    fig, axes = plt.subplots(1, 2)
    sn.kdeplot(data=data, ax=axes[0], x="flakeRate", hue="label", palette=['#d23b2d',"#ffaf00"], common_norm=False, shade=True)
    sn.kdeplot(data=data, ax=axes[1], x="failureRate", hue="label", palette=['#d23b2d',"#ffaf00"], common_norm=False, shade=True)
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[1].set(xlabel="failRate")
    plt.legend(labels=["False alerts", "Legitimate failures"], loc='lower left', bbox_to_anchor=(-0.5, 1.02), ncol=2)
    plt.show()
    return

def checkUsage():
    """
    Check Usage
    """
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage:")
        print("python sandbox.py /path/to/artifactTypes.txt")

if __name__ == "__main__":
    main()
