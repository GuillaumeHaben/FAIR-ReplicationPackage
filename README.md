# FAIR (FAIlure Recognizer)

## Requirements

See `scripts/requirements.txt`

## Install

`git clone git@github.com:GuillaumeHaben/FAIR-ReplicationPackage.git`

`cd FAIR-ReplicationPackage/scripts`

`pip install -r requirements`

## Dataset

Datasets are available at https://figshare.com/projects/Discerning_Legitimate_Failures_From_False_Alerts_A_Study_of_Chromium_s_Continuous_Integration/120852

## Scripts

The following list all the available scripts in `./scripts` used for the study.

## Build Dataset

* Get information about a builder

`python buildDataset.py /PATH/TO/RESULTS BUCKET BUILDER_NAME BUILD_NUMBER NB_BUILDS`

Will get information about the BUILDER_NAME from BUCKET. 
Starts with BUILD_NUMBER, then analyze past NB_BUILDS. 
Save results in /PATH/TO/RESULTS

e.g. `python buildDataset.py ./results ci Mac11.0_Tests 825 2` will get information about tests, build, artifacts for [https://ci.chromium.org/ui/p/chromium/builders/ci/Mac11.0%20Tests/825/test-results?q=](https://ci.chromium.org/ui/p/chromium/builders/ci/Mac11.0%20Tests/825/test-results?q=) and 1 build before (824). Will save the results in `./results`

Important: Space in the builder name should be replaced with `_`

* Results

Results are saved in `./results`. Results about the 2000 builds gathered for the study are saved in `./rawData`.

Folder structure:
```
./results/
    BUCKET.BUILDER_NAME.BUILD_NUMBER/
        testsInfo.json
        buildInfo.json
        1/
            testInfo.json
            Run-ResultId/
                artifacts[.txt|.html]

```

Options for `buildDataset.py`  

- saveArtifacts = True  
Save artifacts for each run if available. Configured to only save [.txt|.html] but it's ready to save [.png|.jpeg] as well.
- notification = True  
Because the scraping crashes sometimes (when too many scripts run in parallel or after a couple of hours), notification sends me a SMS when script is done.
- savePassTests = True  
This option is useful to get information about all tests in 1 build. It will save all ~300k PASS tests. (Useful for `infoBuild.py`)

## Get Test Sources

* Following updates on the platform, this script add test sources for the current commit.
Works for tests which do not contain line number in their metadata (full file test) so mainly `.html` and `.js`. 

`python getSources.py /PATH/TO/RESULTS/BUILDER/`


## Prepare Dataset

* Go through the results folder and prepare a JSON dataset.

`python prepareDataset.py /PATH/TO/RESULTS/BUILDER/`


## Info Dataset

* Get information about a dataset (results folder)

`python infoDataset.py /PATH/TO/RESULTS/BUILDER/`


## Info Runs

* Get more precise information about runs in the dataset

`python infoRuns.py /PATH/TO/RESULTS/testsInfo.json`  


## Model

`python model.py /PATH/TO/RESULTS/dataset.json`

Using `dataset.json`, train and fit a Random Forest Classifier.


### Run properties description

    "buildId": "LuCI build number (specific to builder)",  
    "command": "The command used to run the test",  
    "commandLength": "Number of characters present in the command artifact",  
    "crashlog": "Crash log resulting from an error",  
    "crashlogLength": "Number of characters present in the crashlog artifact",  
    "failureFlipRate": "Failure flip rate considering the previous n commits",  
    "failureRate": "Failure rate considering the previous n commits",  
    "flakeRate": "Flake rate considering the previous n commits",  
    "flakyFlipRate": "Flaky flip rate considering the previous n commits",  
    "label": "0 for a flaky failure (testStatus == 0 && runStatus != 2), 1 for a failure (testStatus == 3)",  
    "runDuration": "Time spent for this run execution",  
    "runStatus": "0: ABORT, 1: FAIL, 2: PASS, 3: CRASH, 4: SKIP",  
    "runTagStatus": "0: CRASH, 1: PASS, 2: FAIL, 3: TIMEOUT, 4: SUCCESS, 5: FAILURE, 6: FAILURE_ON_EXIT, 7: NOTRUN, 8: SKIP, 9: UNKNOWN",  
    "stackTrace": "Stack trace resulting from an error",  
    "stackTraceLength": "Number of characters present in the stackTrace artifact",  
    "stderr": "Stderr captured after the test execution",  
    "stderrLength": "Number of characters present in the stderr artifact",  
    "testId": "Unique identifier for a test",  
    "testSource": "Source code for the test",  
    "testSourceLength": "Number of characters present in the testSource artifact",  
    "testStatus": "0: FLAKY, 1: EXONERATED, 2: EXPECTED, 3: UNEXPECTED, 4: UNEXPECTEDLY_SKIPPED",  
    "testSuite": "Name of the test suite",  
    "testSuiteNumber": "Unique numerical identifier for the test suite"
