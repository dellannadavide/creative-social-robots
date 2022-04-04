from pathlib import Path
from random import Random

import sklearn
from mesa import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy

from simulation.agents.env_agent import EnvironmentAgent
from simulation.agents.patient.fuzzy_patient_ds_agent import FuzzyPatientDSAgent
import utils.constants as Constants
import os

def evalPredictions(X_test, nrrules, y_test_pred, y_test_gold, info_ds, patient):
    accuracy_score = sklearn.metrics.accuracy_score(y_test_gold, y_test_pred)
    feedbacks = patient.getFeedbackForActivitiesinContexts(X_test, y_test_pred, info_ds) #, default_feedback = 0)
    feedback_scores = [numpy.mean(feedbacks), numpy.std(feedbacks), numpy.sum(feedbacks)]

    rep = []
    n = 14
    for i in range(len(y_test_pred)):  # for each activity suggested
        activity = y_test_pred[i]
        count = 0
        for j in range(max(0, i - n), i):
            if y_test_pred[j] == activity:
                count = count + 1
        rep.append(count)
    rep_scores = [numpy.mean(rep), numpy.std(rep), numpy.sum(rep)]

    return [accuracy_score] + [nrrules] + feedback_scores + rep_scores


def evalAll(X_test, y_test_gold, classifiers_predictions, classifiers_rules, info_ds, patient):
    eval_matrix = {}
    for classifier in classifiers_predictions:
        eval_matrix[classifier] = evalPredictions(X_test, classifiers_rules[classifier], classifiers_predictions[classifier], y_test_gold, info_ds,
                                                  patient)
    df = pd.DataFrame.from_dict(eval_matrix, orient='index',
                                    columns=['accuracy', '#(rules)', 'avg feedback', 'std feedback', 'sum feedback',
                                             'avg repetitions', 'std repetitions', 'sum repetitions'])
    print(df)
    return df

def evalAll_dict(X_test, y_test_gold, classifier_predictions, classifier_rules, info_ds, patient):
    return evalPredictions(X_test, classifier_rules, classifier_predictions, y_test_gold, info_ds,
                                                  patient)
    return eval_matrix

def getConsistentData(info_ds_file, y_train, X_test, y_test):
    info_ds = {}
    for info_a in info_ds_file:
        if int(info_a) in y_train:
            info_ds[info_a] = info_ds_file[info_a]
    to_remove = []
    for i in range(len(y_test)):
        if not y_test[i] in y_train:
            to_remove.append(i)
    to_remove = sorted(to_remove, reverse=True)
    for idx in to_remove:
        X_test = numpy.delete(X_test, idx, 0)
        y_test = numpy.delete(y_test, idx, 0)
    return info_ds, X_test, y_test



if __name__ == '__main__':
    """
    This file is used to evaluate the results of the classifiers in terms of accuracy, feedback received from the patient and number of repetitions in the suggestions.
    """
    data_folder = "../simulation/data/survey/"
    results_folder = "../results/classification/"
    use_memory = False
    results_file = results_folder + "RES_benchmarks_s.csv"
    # use_memory = True
    # results_file = results_folder + "RES_benchmarks_m.csv"
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    verbose = Constants.VERBOSE_FALSE

    """ First I gather all required information about the patients and the dataset """
    patients = {}
    datasets_info = {}
    y_test_sets = {}
    X_test_sets = {}
    patients_names = ['90fz82jx', '2ej6b2i7', '5shw42l7', '24i4ipu5', 'ti0bl836', 'x1bislx9', 'pk2bj10v', 'er1qsv6w', '7ei0i2op', 'xglzxsb9', '3ejs2f6p', 'zvwm632u', 'c00gr4wo', 'e33p12vr', 'y4vnuqyu']
    patient_nr = -1
    seed = 15265
    for patient in patients_names:
        paient_nr = patient_nr+1
        dataset_name = data_folder + "DS_" + patient + ".csv"
        info_dataset_name = data_folder + "DS_" + patient + "_info.csv"
        data = pd.read_csv(dataset_name, header=None)
        dataset_size = 5000
        test_ratio = 0.2
        X = data.iloc[:dataset_size, :-1]
        y = data.iloc[:dataset_size, -1:]
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_ratio, random_state=42,
                                                            shuffle=False)
        info = pd.read_csv(info_dataset_name, header=None)
        info_ds_file = {}
        for i in range(len(info.iloc[:])):
            info_ds_file[str(info.iloc[i, 0])] = str(info.loc[i, 1])
        """ Making sure not to have problems with new classes in the test set"""
        info_ds, X_test, y_test = getConsistentData(info_ds_file, y_train, X_test, y_test)

        therapeutic_actions_details_file = data_folder + "therapeutic_actions_"+patient+".csv"
        variables_details_file = data_folder + "variables_survey.csv"
        preferences_details_file = data_folder + "preferences_"+patient+".xlsx"
        """ Preferences file """
        pref_excel_file = pd.ExcelFile(preferences_details_file, engine='openpyxl')
        preference_details = pref_excel_file.parse(sheet_name="Preferences", header=None)
        """ Therapeutic interventions """
        therapeutic_actions_details = pd.read_csv(therapeutic_actions_details_file)
        """ Input-Output control variables """
        variables_details = pd.read_csv(variables_details_file, delimiter=",")
        sim_param = {
            'universes_granularity': 10,
            'patient_name': patient,
            'therapeutic_actions_details_file': therapeutic_actions_details_file,
            'variables_details_file': variables_details_file,
            'preferences_details_file': preferences_details_file
        }
        model = Model()
        env = EnvironmentAgent('Env', variables_details, model, sim_param, verbose)



        memory_info = {}
        if use_memory:
            r = Random()
            r.seed(patient_nr + seed)
            mem_min = 0
            mem_max = 14
            slope_min = 0.01
            slope_max = 0.05
            c = r.randint(mem_min, mem_max)
            b = r.randint(mem_min, c)
            a = r.randint(mem_min, b)
            slope = r.uniform(slope_min, slope_max)
            mood_freq = 0
            memory_info["a"] = a
            memory_info["b"] = b
            memory_info["c"] = c
            memory_info["mem_size"] = mem_max
            memory_info["feedback_change"] = slope
            memory_info["mood_swing_freq"] = mood_freq

        fuzzy_patient = FuzzyPatientDSAgent(patient, model, env, sim_param, therapeutic_actions_details,
                                      variables_details, preference_details, verbose,memory_info=memory_info)

        X_test_sets[patient] = X_test
        y_test_sets[patient] = y_test
        datasets_info[patient] = info_ds
        patients[patient] = fuzzy_patient



    results = []

    for file in os.listdir(results_folder):
        if file.endswith(".xlsx"):
            filepath =  os.path.join(results_folder, file)
            predictions_file = pd.ExcelFile(filepath, engine='openpyxl')
            predictions_sheet = predictions_file.parse(sheet_name="Predictions")
            efs_in_file = predictions_sheet["EFS"].unique()
            for efs in efs_in_file:
                print("Analysing results of classifier "+ efs)
                for ds in datasets_info.keys():
                    pred = predictions_sheet.loc[(predictions_sheet["EFS"] == efs) & (predictions_sheet["Dataset"] == ds), "Prediction"].values
                    int_pred = numpy.array([int(str(s).replace("[[", "").replace("]]", "")) for s in pred])
                    # print(int_pred)
                    # print(patients[ds].actions_centers)
                    if(len(int_pred)>0):
                        print("\tDataset " + ds)
                        results.append([efs, ds] + evalAll_dict(X_test_sets[ds], y_test_sets[ds], int_pred, -1, datasets_info[ds], patients[ds]))
    print(results)
    results_df = pd.DataFrame(results,
                                    columns=['EFS', 'DS', 'accuracy', '#(rules)', 'avg feedback', 'std feedback', 'sum feedback',
                                             'avg repetitions', 'std repetitions', 'sum repetitions'])
    print(results_df)
    results_df.to_csv(results_file)
