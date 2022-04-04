from datetime import datetime
from pathlib import Path
from mesa import Model
import utils.constants as Constants
import itertools as it
import pandas as pd
from simulation.agents.env_agent import EnvironmentAgent
from simulation.agents.patient.fuzzy_patient_ds_agent import FuzzyPatientDSAgent

if __name__ == '__main__':
    """ This file is used to generate the datasets based on the preferences of the survey"""
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
        'universes_granularity': [10],
        'patient_name': ['90fz82jx', '2ej6b2i7', '5shw42l7', '24i4ipu5', 'ti0bl836', 'x1bislx9', 'pk2bj10v', 'er1qsv6w', '7ei0i2op', 'xglzxsb9', '3ejs2f6p', 'zvwm632u', 'c00gr4wo', 'e33p12vr', 'y4vnuqyu'],
        'therapeutic_actions_details_file': [data_folder + "therapeutic_actions_PATIENTNAME.csv"], # leave the _PATIENTNAME suffix, it will be replaced later
        'variables_details_file': [data_folder + "variables_survey.csv"],
        'preferences_details_file': [data_folder + "preferences_PATIENTNAME.xlsx"],
    }

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

        """ Init parameters"""
        sim_param = dict(exp_param)

        """ Create the agents and generate the dataset"""
        env = EnvironmentAgent('Env', variables_details, Model(), sim_param, verbose)
        patient = FuzzyPatientDSAgent(sim_param["patient_name"], Model(), env, sim_param, therapeutic_actions_details,
                                                variables_details, preference_details, verbose)

        print("Generating dataset for patient "+str(exp_param['patient_name'])+"...")
        patient.generateDatasetPreferredActivities(8000, data_folder+"DS_"+str(exp_param['patient_name'])+".csv",
                                                   data_folder+"DS_"+str(exp_param['patient_name'])+"_info.csv")