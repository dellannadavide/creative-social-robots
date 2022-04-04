"""
Module to be invoked via commandline and to be used to run one experiment.
The parameters of the simulation (whose expected types are indicated in the dictionary param_types)
should be given as arguments when invoking the script.
Expected arguments from the script:
<exp_id>, # id of the experiment
<all parameters in param_types>, # simulation parameters
<results_folder>, # path folder where to store the results of the particular experiment
<all_experiments_timestamp>, # timestamp used as an id of the folder containing all experiments run in batch
<verbose>, # whether the simulation should print or not any output to screen
"""

from simulation.run_simulation import *
from utils.utils import *


def main(argv):
    param_types = {
        'nr_interactions': int,
        'nr_steps': int,
        'sampling_trials': int,
        'num_generations': int,
        'nr_rules': int,
        'rate_parents_mating': float,
        'parent_selection_type': str,
        'keep_parents': int,
        'crossover_type': str,
        'mutation_type': str,
        'mutation_probability': float,
        'crossover_probability': int,
        'complexity_cost': float,
        'creativity_gain': float,
        'learning_rate_weights': float,
        'learning_rate_credits': float,
        'default_credit': float,
        'universes_granularity': int,
        'compute_disabled_counterparts_credits': bool,
        'consider_repetition_cost': bool,
        'multiplier_last_n_step_negative_repetition': int,
        'last_n_steps_repetition_halving_cost': float,
        'update_arules': bool,
        'reassess_weights': bool,
        'population_diversity_importance': float,
        'therapies_activities_diversity_importance_ratio': float,
        'patient_type': str,
        'patient_name': str,
        'patient_models_file': str,
        'therapeutic_actions_details_file': str,
        'variables_details_file': str,
        'preferences_details_file': str,
        'trial': int,
        'personalize': bool,
        'noise_probability': float,
        'noise_type': str
    }
    allNames = sorted(param_types)

    if len(argv) != len(param_types) + 4:
        print("ERROR in the input arguments. Given " + str(argv))
        print("Expected <exp_id>, " + str(allNames)+ "<results_folder>, <all_experiments_timestamp>, <verbose>")
        exit()

    """ Decoding the script arguments into parameters of the simulation """
    exp_id = str(datetime.now().strftime("%Y%m%d%H%M%S"))  # + '_' + id_generator()
    exp_param = {}
    exp_param_list = []
    results_folder = ""
    all_experiments_timestamp = ""
    verbose = False
    for arg in argv:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        if k == "exp_id":
            exp_id = exp_id + "_" + str(v)
        if k in param_types.keys():
            if (v == 'True') or (v == 'False'):
                bv = (v == 'True')
                # print(bv)
                exp_param[k] = bv
                exp_param_list.append(bv)
            else:
                exp_param[k] = param_types[k](v)
                exp_param_list.append(param_types[k](v))
        else:
            if k == "results_folder":
                results_folder = str(v)
            if k == "all_experiments_timestamp":
                all_experiments_timestamp = str(v)
            if k == "verbose":
                verbose = int(v)


    patient_pref = None
    """ Patient model """
    if not exp_param["patient_models_file"] == "none":
        patient_models = pd.read_csv(exp_param["patient_models_file"], delimiter=";")
        patient_pref = patient_models[['ta_id', 'sar_action', 'lval', exp_param["patient_name"]]]
    """ Preferences file """
    pref_excel_file = pd.ExcelFile(exp_param["preferences_details_file"].replace("PATIENTNAME", exp_param['patient_name']), engine='openpyxl')
    preference_details = pref_excel_file.parse(sheet_name="Preferences", header=None)


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
    """ --------------------- """

    """ Writing experiment results to file """
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

    writeToFile(results_folder, 'res_' + all_experiments_timestamp + '_' + exp_id, exp_res_dataframes,
                exp_res_dataframes_names)


if __name__ == "__main__":
    main(sys.argv[1:])
