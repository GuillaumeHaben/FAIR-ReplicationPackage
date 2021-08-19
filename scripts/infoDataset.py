import os
import sys
import json
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

def main():
    """
    Main function
    """
    checkUsage()
    
    resultsPath = sys.argv[1]
    countBuildDir = 0

    counter = {
        "FLAKY": 0,
        "EXONERATED": 0,
        "EXPECTED": 0,
        "UNEXPECTED": 0,
        "UNEXPECTEDLY_SKIPPED": 0
    }
    counterRuns = {
        "FLAKY": 0,
        "EXONERATED": 0,
        "EXPECTED": 0,
        "UNEXPECTED": 0,
        "UNEXPECTEDLY_SKIPPED": 0
    }
    dicOccurrence = {
        "FLAKY": {},
        "EXONERATED": {},
        "EXPECTED": {},
        "UNEXPECTED": {},
        "UNEXPECTEDLY_SKIPPED": {}
    }
    durationList = {"FLAKY": [], "UNEXPECTED": []}

    for buildDir in tqdm(sorted(os.scandir(resultsPath), key=lambda e: e.name)):
        countBuildDir += 1
        if os.path.isdir(buildDir) and (int(buildDir.name) < 98177 or int(buildDir.name) > 98192):
            with open(buildDir.path + "/testsInfo.json") as testsInfo:
                data = json.load(testsInfo)
                for test in data:
                    # Count total and unique FLAKY
                    dicOccurrence["FLAKY"], counterFlaky, counterFlakyRuns = countCategory(test, "FLAKY", dicOccurrence["FLAKY"])
                    counter["FLAKY"] += counterFlaky
                    counterRuns["FLAKY"] += counterFlakyRuns
                    # Count total and unique EXONERATED
                    dicOccurrence["EXONERATED"], counterExonerated, counterExoneratedRuns = countCategory(test, "EXONERATED", dicOccurrence["EXONERATED"])
                    counter["EXONERATED"] += counterExonerated
                    counterRuns["EXONERATED"] += counterExoneratedRuns
                    # Count total and unique EXPECTED
                    dicOccurrence["EXPECTED"], counterExpected, counterExpectedRuns = countCategory(test, "EXPECTED", dicOccurrence["EXPECTED"])
                    counter["EXPECTED"] += counterExpected
                    counterRuns["EXPECTED"] += counterExpectedRuns
                    # Count total and unique UNEXPECTED
                    dicOccurrence["UNEXPECTED"], counterUnexpected, counterUnexpectedRuns = countCategory(test, "UNEXPECTED", dicOccurrence["UNEXPECTED"])
                    counter["UNEXPECTED"] += counterUnexpected
                    counterRuns["UNEXPECTED"] += counterUnexpectedRuns
                    # Count total and unique UNEXPECTEDLY_SKIPPED
                    dicOccurrence["UNEXPECTEDLY_SKIPPED"], counterUnexpectedlySkipped, counterUnexpectedlySkippedRuns = countCategory(test, "UNEXPECTEDLY_SKIPPED", dicOccurrence["UNEXPECTEDLY_SKIPPED"])
                    counter["UNEXPECTEDLY_SKIPPED"] += counterUnexpectedlySkipped
                    counterRuns["UNEXPECTEDLY_SKIPPED"] += counterUnexpectedlySkippedRuns
                    # Counting duration for runs
                    for run in test["results"]:
                        if "duration" in run["result"] and run["result"]["status"] != "PASS":
                            duration = float(run["result"]["duration"][:-1])
                            if test["status"] == "FLAKY":
                                durationList["FLAKY"].append(duration)
                            if test["status"] == "UNEXPECTED":
                                durationList["UNEXPECTED"].append(duration)
    
    print("\n--- Results ---")
    print("Number of builds:", countBuildDir)
    print("FLAKY")
    print("    Total:", counter["FLAKY"])
    print("    Unique:", len(dicOccurrence["FLAKY"]))
    print("    Runs:", counterRuns["FLAKY"])
    print("EXONERATED")
    print("    Total:", counter["EXONERATED"])
    print("    Unique:", len(dicOccurrence["EXONERATED"]))
    print("    Runs:", counterRuns["EXONERATED"])
    print("EXPECTED")
    print("    Total:", counter["EXPECTED"])
    print("    Unique:", len(dicOccurrence["EXPECTED"]))
    print("    Runs:", counterRuns["EXPECTED"])
    print("UNEXPECTED")
    print("    Total:", counter["UNEXPECTED"])
    print("    Unique:", len(dicOccurrence["UNEXPECTED"]))
    print("    Runs:", counterRuns["UNEXPECTED"])
    print("UNEXPECTEDLY_SKIPPED")
    print("    Total:", counter["UNEXPECTEDLY_SKIPPED"])
    print("    Unique:", len(dicOccurrence["UNEXPECTEDLY_SKIPPED"]))
    print("    Runs:", counterRuns["UNEXPECTEDLY_SKIPPED"])
    print("Total flaky run duration:", sum(durationList["FLAKY"]))
    print("Total unexpected run duration:", sum(durationList["UNEXPECTED"]))
    

def countCategory(test, category, dicOccurrence):
    counter = 0
    counterRuns = 0
    testId = test["variant"]["def"]["test_suite"] + test["testId"]
    if test["status"] == category:
        counter += 1
        if testId not in dicOccurrence:
            dicOccurrence[testId] = 1
        else:
            dicOccurrence[testId] += 1
        if "results" in test:
            for run in test["results"]:
                if run["result"]["status"] != "PASS":
                    counterRuns += 1
    return dicOccurrence, counter, counterRuns
    
def checkUsage():
    """
    Check Usage
    """
    print("--- Chromium Analysis | Info Dataset ---")
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print("Usage:")
        print("python infoDataset.py /path/to/folder/results/builder/")
        sys.exit(1)

if __name__ == "__main__":
    main()