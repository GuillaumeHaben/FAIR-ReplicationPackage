import os
import sys
import json
import time
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import tree
from pprint import pprint
from datetime import datetime
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from metricUtils import tn, fp, tp, fn, precision, recall, fpr, tpr, tnr, f1, auc, mcc
from sklearn.metrics import make_scorer, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold, train_test_split, RandomizedSearchCV


pd.options.mode.chained_assignment = None  # default='warn'

# Parameters
nbTrees = 100
numWords = 1000
k = 10

def main():
    """
    Main function
    """
    checkUsage()
    startTime = datetime.now()
    
    print("--- Chromium Analysis | Model ---\n")

    # LOAD DATASET
    datasetPath = sys.argv[1]
    # RQ1: ALL ANALYSIS
    data = pd.read_json(datasetPath)

    # RQ2: TEST SUITE ANALYSIS
    # Unit Windows
    # data = data[data["testSuiteNumber"].isin([1, 2, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 41])]
    # Unit Linux
    # data = data[data["testSuiteNumber"].isin([3,5,6,7,8,9,10,11,12,13,18,20,21,22,23,24,25,26,28,29,30,32,34,35,36,37,39,41])]

    # GUI Windows
    # data = data[data["testSuiteNumber"].isin([0,4,15,23])]
    # GUI Linux
    # data = data[data["testSuiteNumber"].isin([0, 1, 2, 14, 15])]

    # Integration / System Windows
    # data = data[data["testSuiteNumber"].isin([3, 7, 11, 18, 20, 29, 34, 39])]
    # Integration / System Linux
    # data = data[data["testSuiteNumber"].isin([4,16,17,27,31,33,38])]
    
    flaky = data[data["label"] == 0]
    failures = data[data["label"] == 1]

    reliableFailures = failures[(failures["flakeRate"] == 0)] 
    unreliableFailures = failures[(failures["flakeRate"] > 0)]

    # General info
    print("Number of flaky runs", len(flaky))
    print("Number of failure runs", len(failures))
    print("Number of reliable failures", len(reliableFailures))
    print("Number of unreliable failures", len(unreliableFailures))

    # Resetting index on subset data and preparing X,y
    data = pd.concat([flaky, unreliableFailures])
    print("Length of dataset of flaky and reliable failures", len(data))
    data.reset_index(drop=True, inplace=True)
    X = data
    y = data['label']

    # PREPARE BAG OF WORDS
    data_text = data["testSource"] + data["stackTrace"] + data["command"] + data["stderr"] + data["crashlog"]
    X_text = X["testSource"] + X["stackTrace"] + X["command"] + X["stderr"] + X["crashlog"]

    # Building Tokenizer and Vocabulary
    tokenizer = Tokenizer(lower=True, num_words=numWords + 1, filters='0123456789\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data_text)

    # Bow features | [:, 1:] see https://github.com/keras-team/keras/issues/8583
    X_bow = tokenizer.texts_to_matrix(X_text, mode="tfidf")[:, 1:]

    # Dynamic features
    X_dynamic = X[['runStatus', 'runDuration', 'runTagStatus', 'stackTraceLength', 
      'commandLength', 'stderrLength', 'crashlogLength', 'testSourceLength']]

    # Merge features in one main array
    X_bow_df = pd.DataFrame(X_bow, columns = list(tokenizer.word_index.keys())[:numWords])
    X_main = X_dynamic.join(X_bow_df)
    
    # SPLITTING DATASET
    X_train, X_test, y_train, y_test = train_test_split(X_main, y, test_size=0.2, stratify=y, shuffle=True)

    # MODEL
    # Create a Classifier for Bag of Words / Artifact analysis and static metrics
    classifier = RandomForestClassifier(n_estimators=nbTrees, verbose=3, n_jobs=10)

    param_distributions = { 
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    CV_rfc = RandomizedSearchCV(estimator=classifier, param_distributions=param_distributions, cv= 5, refit=True)
    CV_rfc.fit(X_train, y_train)

    classifier = CV_rfc.best_estimator_

    print(CV_rfc.best_score_)
    print(CV_rfc.best_params_)

    # ROC and Threshold
    threshold = bestThreshold(classifier, X_test, y_test)

    # Predict the test set
    start = datetime.now()
    y_pred = classifier.predict_proba(X_test)
    end = datetime.now()
    print("Prediction time for", len(y_test), "runs: {}".format(end - start)) 
    y_pred = [1 if y_pred[i][1] > threshold else 0 for i in range(len(y_pred))]

    # Check the metrics
    print("\nMetrics:")
    print("Precision", precision_score(y_test, y_pred))
    print("Recall", recall_score(y_test, y_pred))
    print("MCC", matthews_corrcoef(y_test, y_pred))
    print("F1", f1_score(y_test, y_pred))

    # Analyzing Predicted tests
    # predictionAnalysis(data, y_pred, y_test)

    # Confusion Matrix
    # cm = confusionMatrix(y_test, y_pred)

    # Feature importance
    # featureImportance(classifier, X_main)

    # Visualize a tree
    # visualizeTree(classifier, tokenizer)

    # Logging script execution time
    endTime = datetime.now()
    print('Duration: {}'.format(endTime - startTime))

def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

def bestThreshold(classifier, X_test, y_test):
    # Precision Recall curve threshold
    # predict probabilities
    yhat = classifier.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    yhat = yhat[:, 1]
    # calculate roc curves
    precision, recall, thresholds = precision_recall_curve(y_test, yhat)
    # calculate the g-mean for each threshold
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest g-mean
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    # plot the roc curve for the model
    # no_skill = len(y_test[y_test==1]) / len(y_test)
    # plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    # plt.plot(recall, precision, marker='.', label='Logistic')
    # plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.show()
    bestThreshold = thresholds[ix]
    return bestThreshold

def displayScores(scores, title):
    print("\nMetric: ", title)
    print("Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.nanmean(scores), np.nanstd(scores) * 2))
    return

def predictionAnalysis(data, y_pred, y_test):
    predictions = []
    for i in range(len(y_pred)):
        el = json.loads(data.iloc[y_test.index[i]].to_json())
        if y_pred[i] == 0 and y_test.iloc[i] == 1:
            el["prediciton"] = "FN"
        if y_pred[i] == 1 and y_test.iloc[i] == 1:
            el["prediciton"] = "TP"
        if y_pred[i] == 0 and y_test.iloc[i] == 0:
            el["prediciton"] = "TN"
        if y_pred[i] == 1 and y_test.iloc[i] == 0:
            el["prediciton"] = "FP"
        predictions.append(el)

    saveDataset(predictions, "./predictions.json")
    return

def confusionMatrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Flaky", "Failures"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    return cm

def featureImportance(classifier, X_main):
    # RQ3: Features importance
    print("\nFeatures importance:")
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    for f in range(10):
        if f >= len(X_main.columns):
            break
        print("%d. feature %s (%f)" % (f + 1, X_main.columns[indices[f]], importances[indices[f]]))
    return

def visualizeTree(classifier, tokenizer):
    # print("\n")
    # print(classifier.estimators_[0])

    # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (32,32), dpi=300)
    # cn = ['Flaky', 'Failure']
    # fn = list(tokenizer.word_index.keys())

    # tree.plot_tree(classifier.estimators_[0], 
    # class_names=cn, feature_names=fn)
    # fig.savefig('visTree.png')
    return

def saveDataset(dataset, fileName):
    with open(fileName, 'w') as jsonFile:
        json.dump(dataset, jsonFile, sort_keys=True, indent=4)
    print("File saved to ", fileName)

def displayScoresInline(scores):
    print(
        round(np.nanmean(scores['test_precision']), 2), ", ", 
        round(np.nanmean(scores['test_recall']), 2), ", ", 
        round(np.nanmean(scores['test_f1']), 2), ", ", 
        round(np.nanmean(scores['test_mcc']), 2), 
    sep='')
    print(
        round(np.nanstd(scores['test_precision'] * 2), 2), ", ", 
        round(np.nanstd(scores['test_recall'] * 2), 2), ", ", 
        round(np.nanstd(scores['test_f1'] * 2), 2), ", ", 
        round(np.nanstd(scores['test_mcc'] * 2), 2), 
    sep='')
    return

def checkUsage():
    """
    Check Usage
    """
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage:")
        print("python model.py /path/to/dataset.json")

if __name__ == "__main__":
    main()
