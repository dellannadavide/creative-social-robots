"""
Module to use to run the experiments.
"""

import itertools as it
from pathlib import Path

import memory_profiler

import utils.constants as Constants
from simulation.run_simulation import *
from utils.utils import *

if __name__ == '__main__':
    """
    Step-based experiments.
    Define in the following the possible values of all variables to test
    """

    now = datetime.now()
    all_experiments_timestamp = str(now.strftime("%Y%m%d%H%M%S"))
    data_folder = "../simulation/data/"
    results_folder = "../results/exp_" + all_experiments_timestamp + "/"
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    verbose = Constants.VERBOSE_BASIC

    """ 
    Each of the following parameter values is a list, which can contain more than one value.
    If multiple values are expressed for different parameters, then all combinations of all expressed values are tried
    """
    exp_possible_val = {
        'nr_interactions': [4000],
        'nr_steps': [1],
        'sampling_trials': [1],
        'num_generations': [50],
        'nr_rules': [20],
        'rate_parents_mating': [0.8],
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
        'patient_type': ['random'],
        'patient_name': ['PatientA'],
        'patient_models_file': ['none'],  # to be ignored for the synthetic patients and noise experiments
        'therapeutic_actions_details_file': [data_folder + "therapeutic_actions_PATIENTNAME.csv"],
        # leave the _PATIENTNAME suffix, it will be replaced later
        'variables_details_file': [data_folder + "variables_simple.csv"],
        'preferences_details_file': [data_folder + "preferences_PATIENTNAME.xlsx"],
        # relevant only for experiments not currently reported in the paper
        'personalize': [True],
        'noise_probability': [0],  # should be in [0,1]
        'noise_type': ['none'],  # can be 'gaussian', 'inv_gaussian', 'reversed_feedback',
        'trial': list(range(200))  # can also be just one value [0]
    }

    """
    Here all the parameters expressed above are combined.
    Each of the possible combination is an experiment.
    Note that the parameter "trial" is included here. 
    By combining trial with the other parameters we obtain the repeated trials
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
        pref_excel_file = pd.ExcelFile(exp_param["preferences_details_file"].replace("PATIENTNAME", exp_param['patient_name']), engine='openpyxl')
        preference_details = pref_excel_file.parse(sheet_name="Preferences", header=None)
        """ Patient model IF ANY (NOT USED NOW)"""
        patient_pref = None
        if not exp_param["patient_models_file"]=="none":
            patient_models = pd.read_csv(exp_param["patient_models_file"], delimiter=";")
            patient_pref = patient_models[['ta_id', 'sar_action', 'lval', exp_param["patient_name"]]]
        """ Therapeutic interventions """
        therapeutic_actions_details = pd.read_csv(exp_param["therapeutic_actions_details_file"].replace("PATIENTNAME", exp_param['patient_name']))
        """ Input-Output control variables """
        variables_details = pd.read_csv(exp_param["variables_details_file"], delimiter=",")

        """ RUNNING THE SIMULATION """
        exp_time, rules_credits, rules, actions_credits, predicted_happiness, \
        measured_happiness, actual_happiness, forecast_errors, diversity_therapies, covered_therapies_ratio, \
        diversity_activities, covered_activities_ratio, equal_to_last_interaction, \
        suggested_activities, suggested_modalities, contexts, context_names = \
            runSimulation(patient_pref, therapeutic_actions_details, variables_details, exp_param, verbose,
                          pref_details=preference_details)

        """ Writing current simulation results to file """
        exp_res_dataframes, exp_res_dataframes_names = transformResultsIntoDataFrames(allNames, exp_param,
                                                                                      exp_param_list, exp_time,
                                                                                      rules_credits, rules,
                                                                                      actions_credits,
                                                                                      predicted_happiness,
                                                                                      measured_happiness,
                                                                                      actual_happiness,
                                                                                      forecast_errors,
                                                                                      diversity_therapies,
                                                                                      covered_therapies_ratio,
                                                                                      diversity_activities,
                                                                                      covered_activities_ratio,
                                                                                      equal_to_last_interaction,
                                                                                      suggested_activities,
                                                                                      suggested_modalities, contexts,
                                                                                      context_names)

        exp_id = str(datetime.now().strftime("%Y%m%d%H%M%S")) + '_' + id_generator()
        writeToFile(results_folder, 'res_' + all_experiments_timestamp + '_' + exp_id, exp_res_dataframes,
                    exp_res_dataframes_names)
