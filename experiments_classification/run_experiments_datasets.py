"""
Module to use to run the experiments.
"""

import itertools as it
from pathlib import Path

import memory_profiler
from mesa import Model
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

import utils.constants as Constants
from simulation.agents.env_agent import EnvironmentAgent
from simulation.agents.patient.fuzzy_patient_ds_agent import FuzzyPatientDSAgent
from simulation.agents.sar.sar_agent import SARAgent
from simulation.run_simulation import *
from utils.utils import *

from sklearn.model_selection import train_test_split
import numpy
import scipy.io
import pandas

pandas.set_option("display.max_rows", None, "display.max_columns", None)
import math
import os
import sklearn.metrics
from benchmark.almmo1.ALMMo1_System import ALMMo1_classification_learning
# Import training function of ALMMo-1 system for classification
from benchmark.almmo1.ALMMo1_System import ALMMo1_classification_testing
# Import function of the ALMMo-1 system for classification during validation stage

import matlab.engine


def evalPredictions(X_test, nrrules, y_test_pred, y_test_gold, info_ds, patient):
    accuracy_score = sklearn.metrics.accuracy_score(y_test_gold, y_test_pred)
    feedbacks = patient.getFeedbackForActivitiesinContexts(X_test, y_test_pred, info_ds)
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
    df = pandas.DataFrame.from_dict(eval_matrix, orient='index',
                                    columns=['accuracy', '#(rules)', 'avg feedback', 'std feedback', 'sum feedback',
                                             'avg repetitions', 'std repetitions', 'sum repetitions'])
    print(df)
    return df


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
    now = datetime.now()
    all_experiments_timestamp = str(now.strftime("%Y%m%d%H%M%S"))
    data_folder = "../simulation/data/survey/"
    results_folder = "../results/exp_" + all_experiments_timestamp + "/"
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    verbose = Constants.VERBOSE_FALSE

    """ 
        Each of the following parameter values is a list, which can contain more than one value.
        If multiple values are expressed for different parameters, then all combinations of all expressed values are tried
        """
    exp_possible_val = {
        'nr_interactions': [4000],
        'nr_steps': [1],
        'sampling_trials': [10],
        'num_generations': [200],
        'nr_rules': [20],
        'rate_parents_mating': [0.5],
        'parent_selection_type': ['sss'],
        'keep_parents': [1],
        'crossover_type': ['uniform'],
        'mutation_type': ['random'],
        'mutation_probability': [0.25],
        'crossover_probability': [1],
        'complexity_cost': [0],  # todo currently not supported -> ignore
        'creativity_gain': [0],  # todo currently not supported -> ignore
        'learning_rate_weights': [0.01],
        'learning_rate_credits': [0.04],  # should be >0, with 0 the system will not learn
        'default_credit': [1],
        'universes_granularity': [10],
        'compute_disabled_counterparts_credits': [False],  # todo currently not supported -> ignore
        'consider_repetition_cost': [True],
        'multiplier_last_n_step_negative_repetition': [7],  # 2,7
        'last_n_steps_repetition_halving_cost': [10],  # 0,2,10
        'update_arules': [False],  # todo currently not supported -> ignore
        'reassess_weights': [False],  # todo currently not supported -> ignore
        'population_diversity_importance': [0.5],  # should be in [0,1]
        'therapies_activities_diversity_importance_ratio': [0.75],  # should be in [0,1]
        'patient_type': ['fuzzy_ds'],
        'patient_name': ['90fz82jx', '2ej6b2i7', '5shw42l7', '24i4ipu5', 'ti0bl836', 'x1bislx9', 'pk2bj10v', 'er1qsv6w', '7ei0i2op', 'xglzxsb9', '3ejs2f6p', 'zvwm632u', 'c00gr4wo', 'e33p12vr', 'y4vnuqyu'],
        # 'patient_name': ['y4vnuqyu'],
        'therapeutic_actions_details_file': [data_folder + "therapeutic_actions_PATIENTNAME.csv"],
        # leave the _PATIENTNAME suffix, it will be replaced later
        'variables_details_file': [data_folder + "variables_survey.csv"],
        'preferences_details_file': [data_folder + "preferences_PATIENTNAME.xlsx"],
        'personalize': [True],
        'noise_probability': [0],  # should be in [0,1]
        'noise_type': ['none'],  # can be 'gaussian', 'inv_gaussian', 'reversed_feedback',
        'trial': [0]  # can also be just one value [0]
    }

    exp_res = []
    """
    Here all the parameters expressed above are combined.
    Each of the possible combination is an experiment.
    """
    allNames = sorted(exp_possible_val)
    combinations = it.product(*(exp_possible_val[Name] for Name in allNames))
    for c in combinations:
        exp_param = {}
        exp_param_list = []
        for i in range(len(allNames)):
            exp_param[allNames[i]] = c[i]
            exp_param_list.append(c[i])

        """ Preferences file """
        pref_excel_file = pd.ExcelFile(
            exp_param["preferences_details_file"].replace("PATIENTNAME", exp_param['patient_name']), engine='openpyxl')
        preference_details = pref_excel_file.parse(sheet_name="Preferences", header=None)
        """ Therapeutic interventions """
        therapeutic_actions_details = pd.read_csv(
            exp_param["therapeutic_actions_details_file"].replace("PATIENTNAME", exp_param['patient_name']))
        """ Input-Output control variables """
        variables_details = pd.read_csv(exp_param["variables_details_file"], delimiter=",")
        sim_param = dict(exp_param)
        sim_param["num_parents_mating"] = math.ceil(exp_param["nr_rules"] * exp_param["rate_parents_mating"])
        sim_param["sol_per_pop"] = exp_param["nr_rules"]
        sim_param["last_n_steps_negative_repetition"] = exp_param["nr_steps"] * exp_param[
            "multiplier_last_n_step_negative_repetition"]

        print(str(exp_param['patient_name']))

        results_file = results_folder + "RES_DS_" + str(exp_param['patient_name']) + ".csv"
        dataset_name = data_folder + "DS_" + str(exp_param['patient_name']) + ".csv"
        info_dataset_name = data_folder + "DS_" + str(exp_param['patient_name']) + "_info.csv"
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

        datain_tr = numpy.matrix(X_train)  # Input
        dataout_tr = numpy.matrix(y_train)  # The labels of the respective data samples
        datain_te = numpy.matrix(X_test)
        dataout_te = numpy.matrix(y_test)

        # print(info_ds)
        attributes = [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [1, len(info_ds)]]
        inputs = ['Weather', 'TimeDay', 'TypeDay']
        outputs = ['Class']

        """ Create corresponding dataset for the matlab libraries """
        mat_filename = data_folder + "mat/DS_" + str(exp_param["patient_name"]) + ".mat"
        scipy.io.savemat(mat_filename,
                         mdict={'DTra1': numpy.matrix(X_train), 'LTra1': numpy.matrix(y_train.astype(float)),
                                'DTes1': numpy.matrix(X_test), 'LTes1': numpy.matrix(y_test.astype(float))})


        """ Create the agents"""
        model = Model()
        env = EnvironmentAgent('Env', variables_details, model, sim_param, verbose)
        patient = FuzzyPatientDSAgent(sim_param["patient_name"], model, env, sim_param, therapeutic_actions_details,
                                      variables_details, preference_details, verbose)

        """ Defines the systems to compare """
        systems_to_compare = {"max-acc", "random", "ALLMo0", "ALLMo1", "SVM", "SAFLS", "PSOALFS", "FWAadaBoostSOFIES",
                              "skmoefs-MPAES_RCS", "MEEFIS", "ETS", "SAR",
                              "ANFIS-GA", "ANFIS-PSO"} #all implemented
        systems_to_compare = {"max-acc", "random", "ALLMo0", "SAFLS", "PSOALFS", "FWAadaBoostSOFIES",
                              "skmoefs-MPAES_RCS", "MEEFIS", "ETS", "SAR",
                              "ANFIS-GA", "ANFIS-PSO"} #only interesting ones
        systems_to_compare = {"max-acc", "random", "ALLMo0", "SAFLS", "PSOALFS", "FWAadaBoostSOFIES",
                              "skmoefs-MPAES_RCS", "MEEFIS", "ETS",
                              "ANFIS-GA", "ANFIS-PSO"} #all beanchmarks (i.e., no SAR)
        systems_to_compare = {"ANFIS-GA"}  # all beanchmarks (i.e., no SAR) apart from anfis
        # systems_to_compare = {"ANFIS-PSO"}

        test_predictions = {}
        nr_rules = {}

        name = "max-acc"
        if name in systems_to_compare:
            print(name)
            pred = numpy.matrix(y_test)
            print(sklearn.metrics.accuracy_score(numpy.matrix(y_test), pred))
            test_predictions[name] = pred
            nr_rules[name] = 0
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)

        name = "random"
        if name in systems_to_compare:
            print(name)
            pred = []
            for i in y_test:
                pred.append(random.randint(int(attributes[-1][0]), int(attributes[-1][1])))
            print(sklearn.metrics.accuracy_score(numpy.matrix(y_test), pred))
            test_predictions[name] = pred
            nr_rules[name] = 0
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)

        name = "SAR"
        if name in systems_to_compare:
            filename = 'genetic' + str(now.strftime("%y%m%d%H%M%S"))
            # print(name + "-evol-learn_onlytrain")
            # nao = SARAgent('Nao', Model(), patient, env, filename, sim_param, therapeutic_actions_details,
            #                variables_details, verbose,
            #                pref_details=preference_details, classifier_mode=True)
            # rules_details, nao_est_training = nao.train(X_train, y_train, info_ds, evolution=False)
            # nao_est_test = nao.test(X_test, y_test, info_ds, evolution=True, initial_rules_details=rules_details,
            #                         update_credits=False)
            # print(sklearn.metrics.accuracy_score(numpy.matrix(y_test), nao_est_test))
            # test_predictions[name + "-evol-learn_onlytrain"] = nao_est_test
            # nr_rules[name + "-evol-learn_onlytrain"] = sim_param["nr_rules"]
            # evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
            print(name + "-evol-learn_alsotest")
            nao = SARAgent('Nao', Model(), patient, env, filename, sim_param, therapeutic_actions_details,
                           variables_details, verbose,
                           pref_details=preference_details, classifier_mode=True)
            rules_details, nao_est_training = nao.train(X_train, y_train, info_ds, evolution=False)
            nao_est_test = nao.test(X_test, y_test, info_ds, evolution=True, initial_rules_details=rules_details, update_credits=True)
            print(sklearn.metrics.accuracy_score(numpy.matrix(y_test), nao_est_test))
            test_predictions[name+"-evol-learn_alsotest"] = nao_est_test
            nr_rules[name+"-evol-learn_alsotest"] = sim_param["nr_rules"]
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)

        name = "ALLMo0"
        if name in systems_to_compare:
            print(name)
            from benchmark.almmo0.ALMMo0_System import ALMMo0classifier_testing
            from benchmark.almmo0.ALMMo0_System import ALMMo0classifier_learning

            ## read the dataset
            mat_contents = scipy.io.loadmat(mat_filename)
            tradata = mat_contents['DTra1']
            tralabel = mat_contents['LTra1']
            tesdata = mat_contents['DTes1']
            teslabel = mat_contents['LTes1']
            # Training
            SystemParam = ALMMo0classifier_learning(tradata, tralabel)  # Train the ALMMo-1 system
            # Validation
            estimation = ALMMo0classifier_testing(tesdata, SystemParam)  # Validate the trained ALMMo-1 system
            # print(estimation)
            # print(sklearn.metrics.confusion_matrix(dataout, estimation))  # Calculate the confusion matrix
            print(sklearn.metrics.accuracy_score(teslabel, estimation))  # Calculate the confusion matrix
            test_predictions[name] = estimation
            nr_rules[name] = SystemParam["Class_number"]
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)

        name = "ALLMo1"
        if name in systems_to_compare:
            print(name)
            # Training
            Estimation, SystemParam = ALMMo1_classification_learning(datain_tr, dataout_tr)  # Train the ALMMo-1 system
            # Validation
            estimation = ALMMo1_classification_testing(datain_te, SystemParam)  # Validate the trained ALMMo-1 system
            # print(estimation)
            # print(sklearn.metrics.confusion_matrix(dataout, estimation))  # Calculate the confusion matrix
            print(sklearn.metrics.accuracy_score(dataout_te, estimation))  # Calculate the confusion matrix
            test_predictions[name] = estimation
            nr_rules[name] = SystemParam["ModeNumber"]
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)

        name = "SVM"
        if name in systems_to_compare:
            print(name)
            from sklearn.svm import SVC

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            clf.fit(datain_tr, dataout_tr)
            Pipeline(steps=[('standardscaler', StandardScaler()),
                            ('svc', SVC(gamma='auto'))])
            # print(clf.score(X_test, y_test))
            est_svm = clf.predict(X_test)
            # print(sklearn.metrics.confusion_matrix(dataout, est_svm))  # Calculate the confusion matrix
            print(sklearn.metrics.accuracy_score(dataout_te, est_svm))  # Calculate the confusion matrix
            test_predictions[name] = est_svm
            nr_rules[name] = 0
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)

        name = "SAFLS"
        if name in systems_to_compare:
            print(name)
            eng = matlab.engine.start_matlab()
            safls_folder = eng.genpath('benchmark/safls')
            eng.addpath(safls_folder, nargout=0)
            eng.workspace['DSName'] = mat_filename
            eng.RunSAFLS(nargout=0)
            output_labels = eng.eval('label_est')
            nr_rules[name] = int(eng.eval('nrrules'))
            eng.quit()
            # print(output_labels)
            est_safls = numpy.matrix(output_labels).astype(int)
            # print(sklearn.metrics.confusion_matrix(dataout, est_safls))  # Calculate the confusion matrix
            print(sklearn.metrics.accuracy_score(dataout_te, est_safls))  # Calculate the confusion matrix
            test_predictions[name] = est_safls
            evalAll(X_test, dataout_te, test_predictions, nr_rules,  info_ds, patient)
            time.sleep(5)

        name = "PSOALFS"
        if name in systems_to_compare:
            print(name)
            eng = matlab.engine.start_matlab()
            psoalfs_folder = eng.genpath('benchmark/psoalfs')
            eng.addpath(psoalfs_folder, nargout=0)
            eng.workspace['DSName'] = mat_filename
            eng.RunPSOALFS(nargout=0)
            output_labels = eng.eval('Prediction')
            eng.quit()
            # print(output_labels)
            est_psoalfs = numpy.matrix(output_labels)
            est_psoalfs = numpy.around(est_psoalfs, 0).astype(int)
            # print(est_psoalfs)
            # print(sklearn.metrics.confusion_matrix(dataout, est_psoalfs))
            print(sklearn.metrics.accuracy_score(dataout_te, est_psoalfs))  # Calculate the confusion matrix
            test_predictions[name] = est_psoalfs
            nr_rules[name] = len(info_ds)
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
            time.sleep(5)

        name = "FWAadaBoostSOFIES"
        if name in systems_to_compare:
            print(name)
            eng = matlab.engine.start_matlab()
            fwadaboostsofies_folder = eng.genpath('benchmark/fwadaboostsofies')
            eng.addpath(fwadaboostsofies_folder, nargout=0)
            eng.workspace['DSName'] = mat_filename
            eng.RunFWAdaBoostSOFIES(nargout=0)
            output_labels = eng.eval('PredFAS')
            eng.quit()
            # print(output_labels)
            est_fwadaboostsofies = numpy.matrix(output_labels)
            est_fwadaboostsofies = numpy.around(est_fwadaboostsofies, 0).astype(int)
            # print(est_psoalfs)
            # print(sklearn.metrics.confusion_matrix(dataout, est_psoalfs))
            print(sklearn.metrics.accuracy_score(dataout_te, est_fwadaboostsofies))  # Calculate the confusion matrix
            test_predictions[name] = est_fwadaboostsofies
            nr_rules[name] = len(info_ds)
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
            time.sleep(5)

        name = "skmoefs-MPAES_RCS"
        if name in systems_to_compare:
            print(name)
            from platypus.algorithms import *
            from benchmark.skmoefs.toolbox import MPAES_RCS, load_dataset, normalize
            from benchmark.skmoefs.rcs import RCSInitializer, RCSVariator
            from benchmark.skmoefs.discretization.discretizer_base import fuzzyDiscretization
            X_n, y_n = normalize(X.values, y.iloc[:, 0].values, attributes)
            Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=test_ratio, random_state=42, shuffle=False)
            info_ds, Xte, yte = getConsistentData(info_ds_file, ytr, Xte, yte)
            my_moefs = MPAES_RCS(variator=RCSVariator(), initializer=RCSInitializer())
            my_moefs.fit(Xtr, ytr, max_evals=10000)
            moefs_output = my_moefs.predict(Xte)
            moefs_output = list(moefs_output.astype(int))
            print(sklearn.metrics.accuracy_score(numpy.matrix(y_test), moefs_output))
            test_predictions[name] = moefs_output
            nr_rules[name] = len(my_moefs.classifiers[0].rules)
            # my_moefs.show_model()
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
            time.sleep(5)

        name = "ANFIS-GA"
        if name in systems_to_compare:
            print(name)
            eng = matlab.engine.start_matlab()
            anfis_folder = eng.genpath('benchmark/anfis')
            eng.addpath(anfis_folder, nargout=0)
            eng.workspace['DSName'] = mat_filename
            eng.workspace['nclasses'] = len(info_ds)
            eng.RunANFISGA(nargout=0)
            output_labels = eng.eval('TestOutputs')
            eng.quit()
            est_anfisga = numpy.matrix(output_labels)
            print(est_anfisga)
            est_anfisga = numpy.around(est_anfisga, 0).astype(int)
            for e in range(len(est_anfisga)):
                if est_anfisga[e]>len(info_ds):
                    est_anfisga[e] = len(info_ds)
                if est_anfisga[e] < 1:
                    est_anfisga[e] = 1
            print(est_anfisga)
            # print(sklearn.metrics.confusion_matrix(dataout, est_psoalfs))
            print(sklearn.metrics.accuracy_score(dataout_te, est_anfisga))  # Calculate the confusion matrix
            test_predictions[name] = est_anfisga
            nr_rules[name] = len(info_ds)
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
            time.sleep(5)

        name = "ANFIS-PSO"
        if name in systems_to_compare:
            print(name)
            eng = matlab.engine.start_matlab()
            anfis_folder = eng.genpath('benchmark/anfis')
            eng.addpath(anfis_folder, nargout=0)
            eng.workspace['DSName'] = mat_filename
            eng.workspace['nclasses'] = len(info_ds)
            eng.RunANFISPSO(nargout=0)
            output_labels = eng.eval('TestOutputs')
            eng.quit()
            # print(output_labels)
            est_anfispso = numpy.matrix(output_labels)
            est_anfispso = numpy.around(est_anfispso, 0).astype(int)
            for e in range(len(est_anfispso)):
                if est_anfispso[e]>len(info_ds):
                    est_anfispso[e] = len(info_ds)
                if est_anfispso[e] < 1:
                    est_anfispso[e] = 1
            # print(est_psoalfs)
            # print(sklearn.metrics.confusion_matrix(dataout, est_psoalfs))
            print(sklearn.metrics.accuracy_score(dataout_te, est_anfispso))  # Calculate the confusion matrix
            test_predictions[name] = est_anfispso
            nr_rules[name] = len(info_ds)
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
            time.sleep(5)

        name = "MEEFIS"
        if name in systems_to_compare:
            print(name)
            eng = matlab.engine.start_matlab()
            meefis_folder = eng.genpath('benchmark/meefis')
            eng.addpath(meefis_folder, nargout=0)
            eng.workspace['DSName'] = mat_filename
            eng.RunMEEFIS(nargout=0)
            output_labels = eng.eval('Output')
            totnrules = eng.eval('totnrules')
            avgnrules = eng.eval('avgnrules')
            eng.quit()
            # print(output_labels)
            est_meefis = numpy.matrix(output_labels)
            est_meefis = numpy.around(est_meefis, 0).astype(int)
            # print(est_psoalfs)
            # print(sklearn.metrics.confusion_matrix(dataout, est_psoalfs))
            print(sklearn.metrics.accuracy_score(dataout_te, est_meefis))  # Calculate the confusion matrix
            test_predictions[name] = est_meefis
            nr_rules[name] = totnrules
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
            time.sleep(5)

        name = "ETS"
        if name in systems_to_compare:
            print(name)
            eng = matlab.engine.start_matlab()
            ets_folder = eng.genpath('benchmark/ets_refactored')
            eng.addpath(ets_folder, nargout=0)
            eng.workspace['DSName'] = mat_filename
            eng.RunETS(nargout=0)
            output_labels = eng.eval('est')
            nr_clusters = eng.eval('max_nr_clusters')
            eng.quit()
            est_ets = numpy.matrix(output_labels)
            est_ets = numpy.around(est_ets, 0).astype(int)
            for i in range(len(est_ets)):
                if est_ets[i]<=0:
                    est_ets[i] = 1
                else:
                    if est_ets[i]>max(dataout_te):
                        est_ets[i] = max(dataout_te)
            i=0
            missing_nr = len(dataout_te) - len(est_ets)
            for i in range(missing_nr):
                missing = [[dataout_te[len(est_ets)-(missing_nr-i+1), 0]]]
                est_ets=numpy.append(est_ets, missing, axis=0)

            # print(est_psoalfs)
            # print(sklearn.metrics.confusion_matrix(dataout_te, est_ets))
            print(sklearn.metrics.accuracy_score(dataout_te, est_ets))  # Calculate the confusion matrix
            test_predictions[name] = est_ets
            nr_rules[name] = nr_clusters
            evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
            time.sleep(5)

        df_eval_all = evalAll(X_test, dataout_te, test_predictions, nr_rules, info_ds, patient)
        df_eval_all.to_csv(results_file)

        for s in systems_to_compare:
            for i in range(len(test_predictions[s])):
                exp_res.append(
                    [s, exp_param['patient_name'], i,  test_predictions[s][i]])

    print("Writing predictions to file...")
    df_res = pd.DataFrame(exp_res,
                          columns=["EFS", "Dataset", "Index DataPoint", "Prediction"])
    writeToFile(results_folder, 'res_' + all_experiments_timestamp + '_' + 'benchmarks', [df_res], ["Predictions"])
