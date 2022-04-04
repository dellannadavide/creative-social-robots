import math
from datetime import datetime
from pathlib import Path

import numpy
import pandas as pd
from mesa import Model

from simulation.agents.env_agent import EnvironmentAgent
from simulation.agents.patient.fuzzy_patient_ds_agent import FuzzyPatientDSAgent
from simulation.agents.sar.sar_agent import SARAgent
import utils.constants as Constants

def evalSAROnDataset(X_train, y_train, X_test, y_test, info_ds, patient_pref, therapeutic_actions_details, variables_details, exp_param, verbose, preference_details=None):
    """
     This function is used to evaluate the SAR agent, implementing the EFSAR algorithm on a dataset.
     It takes a training and test set and returns the predictions for the test set and the number of rules used during the process.
    """
    sim_param = dict(exp_param)
    sim_param["num_parents_mating"] = math.ceil(exp_param["nr_rules"] * exp_param["rate_parents_mating"])
    sim_param["sol_per_pop"] = exp_param["nr_rules"]
    sim_param["last_n_steps_negative_repetition"] = exp_param["nr_steps"] * exp_param[
        "multiplier_last_n_step_negative_repetition"]
    model = Model()
    env = EnvironmentAgent('Env', variables_details, model, sim_param, verbose)
    patient = FuzzyPatientDSAgent(sim_param["patient_name"], model, env, sim_param, therapeutic_actions_details,
                                  variables_details, preference_details, verbose)
    now = datetime.now()
    filename = 'efsar_' + str(now.strftime("%y%m%d%H%M%S"))
    nao = SARAgent('Nao', Model(), patient, env, filename, sim_param, therapeutic_actions_details,
                   variables_details, verbose,
                   pref_details=preference_details, classifier_mode=True)
    rules_details, nao_est_training = nao.train(X_train, y_train, info_ds, evolution=False)
    nao_est_test = nao.test(X_test, y_test, info_ds, evolution=True, initial_rules_details=rules_details, update_credits=True)
    return nao_est_test, sim_param["nr_rules"]
