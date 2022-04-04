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
from sklearn.model_selection import train_test_split

from eval_sar_on_dataset import evalSAROnDataset
from simulation.run_simulation import *
from utils.utils import *



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
    therapeutic_actions_details = pd.read_csv(exp_param["therapeutic_actions_details_file"].replace("PATIENTNAME", exp_param['patient_name']),encoding= 'unicode_escape')
    """ Input-Output control variables """
    variables_details = pd.read_csv(exp_param["variables_details_file"], delimiter=",")

    data_folder = "../simulation/data/survey/"
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
    """ RUNNING THE SIMULATION """
    predictions, rules = \
        evalSAROnDataset(X_train, y_train, X_test, y_test, info_ds, patient_pref, therapeutic_actions_details, variables_details, exp_param, verbose,
                      preference_details=preference_details)
    """ --------------------- """

    """ Writing experiment results to file """
    exp_res = []
    for i in range(len(predictions)):
        exp_res.append(
            ["SAR", exp_param['patient_name'], i, predictions[i]])
    df_res = pd.DataFrame(exp_res,
                                columns=["EFS", "Dataset", "Index DataPoint", "Prediction"])
    writeToFile(results_folder, 'res_' + all_experiments_timestamp + '_' + exp_id, [df_res], ["Predictions"])


if __name__ == "__main__":
    main(sys.argv[1:])
