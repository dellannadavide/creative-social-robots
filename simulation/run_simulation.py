""" Module containing functions to be used to run a simulation """

# import cProfile
# import io
# import os
# import pstats
from datetime import datetime
import time
# from pstats import SortKey

from simulation.interactionmodel import NaoInteractionModel
from simulation.utils.utils import *
import math
import re

import utils.constants as Constants


def runSimulation(patient_pref, therapeutic_actions_details, variables_details, exp_param, verbose, pref_details=None):
    """
    Given a dictionary of simulation parameters @sim_param, this function
    - Creates a MAS with the three types of agents (environment, patient, and SAR),
    - Progresses the MAS step by step (executing each agent in the order above)
    - After every step asks the SAR store the knowledge about the step just concluded
    - After every interaction (i.e., a number of steps), asks the SAR to run the creative personlization module,
    and asks every agent to prepare for the next interaction
    - Once the simulation is over (i.e., all interactions have been run), it collects all needed info for logging of the
    results
    """

    """ Init parameters"""
    sim_param = dict(exp_param)
    sim_param["num_parents_mating"] = math.ceil(exp_param["nr_rules"] * exp_param["rate_parents_mating"])
    sim_param["sol_per_pop"] = exp_param["nr_rules"]
    sim_param["last_n_steps_negative_repetition"] = exp_param["nr_steps"] * exp_param[
        "multiplier_last_n_step_negative_repetition"]

    """ Timestamp """
    now = datetime.now()
    filename = 'genetic' + str(now.strftime("%y%m%d%H%M%S"))

    """ Create the MAS """
    start_time = time.time()
    if verbose == Constants.VERBOSE_BASIC:
        print("==== Initialization ====\n")
    model = NaoInteractionModel(1, filename, sim_param, patient_pref, therapeutic_actions_details, variables_details, #todo n.b. patient_pref only used for the base case (not used now in the experiments)
                                verbose, pref_details=pref_details)
    if verbose == Constants.VERBOSE_BASIC:
        print("INITIALIZATION TIME ELAPSED --- %s seconds ---" % (time.time() - start_time))
        print("========================\n")
        print("\n")

    """ Run the Simulation """
    curr_int = 0
    for interaction in range(sim_param["nr_interactions"]):
        interaction_time = time.time()
        # interaction phase
        curr_int = interaction

        if verbose == Constants.VERBOSE_BASIC:
            print("==== " + str(interaction) + "-th interaction with patient ====")
        for i in range(sim_param["nr_steps"]):
            # pr = cProfile.Profile()
            # pr.enable()
            model.step()
            # pr.disable()
            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # os.system('cls' if os.name == 'nt' else 'clear')
            # ps.print_stats(10)
            # print(s.getvalue())
            model.schedule.agents[1].collectKnowledge()

        if verbose == Constants.VERBOSE_BASIC:
            print("=== End " + str(interaction) + "-th interaction with patient ====\n")
            print("=== Personalization Phase ===")
            print("... Analyzing information from " + str(interaction) + "-th interaction ...")
        model.schedule.agents[1].personalizer.run()
        # model.schedule.agents[1].printRules()
        # if discard:
        #   raise ValueError('no creative rule could be synthesised')
        if verbose == Constants.VERBOSE_BASIC:
            print("... Preparing for new interaction ...\n")
        model.schedule.agents[1].prepareForNewInteraction()
        model.schedule.agents[2].prepareForNewInteraction()
        if verbose == Constants.VERBOSE_BASIC:
            print("INTERACTION TIME ELAPSED --- %s seconds ---" % (time.time() - interaction_time))
            print("============================\n")

    exp_time = time.time() - start_time

    """ Collect, aggregate and store simulation data for logging and results analysis """
    """ Data about the credits of rules with output a certain action """
    actions_credits = {}
    for sr in model.schedule.agents[1].rules_credits:
        action = "a_" + str(re.search("a_(.*?)\]", sr).group(1)) + "]"
        credit = model.schedule.agents[1].rules_credits[sr]
        if action in actions_credits:
            actions_credits[action].append(credit)
        else:
            actions_credits[action] = [credit]

    """ Data about the feedback (predicted, measured, and actual) of the patient """
    predicted_happiness = []
    measured_happiness = []
    actual_happiness = []
    for k in model.schedule.agents[1].logInteractions:
        predicted_happiness.append(k["expected_emotions"])
        measured_happiness.append(k["detected_emotions"])
        actual_happiness.append(k["emotions"])
    forecast_errors = [measured_happiness[i] - predicted_happiness[i] for i in range(len(measured_happiness))]

    """ Data about the diversity indeces throughout the simulation"""
    diversity_therapies = []
    diversity_activities = []
    # max_diversity_rules = []
    covered_therapies_ratio = []
    covered_activities_ratio = []
    for k in model.schedule.agents[1].logInteractions:
        diversity_index_therapies, individuals_per_species_therapies, \
        diversity_index_activities, individuals_per_species_activities = \
            computeRuleBaseDiverstity(model.schedule.agents[1].actions_to_ti, model.schedule.agents[1].ta, k["rulebase"])
        diversity_therapies.append(diversity_index_therapies)
        diversity_activities.append(diversity_index_activities)
        # diversity_rules.append(diversity_index[0])
        # max_diversity_rules.append(diversity_index[1])
        covered_therapies_ratio.append(len(individuals_per_species_therapies) / len(model.schedule.agents[1].ta))
        covered_activities_ratio.append(len(individuals_per_species_activities) / len(model.schedule.agents[1].action_var))

    """ Data about the linguistic interpretation of the contexts at every step """
    contexts_names = []
    contexts = []
    suggested_activities = []
    suggested_modalities = []
    last_step = []
    curr_step = []
    equal_to_last_interaction = 0
    s = 0
    for k in model.schedule.agents[1].logInteractions:
        contexts_names = sorted(k["context"])
        contexts.append([k["context"][c] for c in contexts_names])
        if len(k["sugg"]) > 0:
            curr_step.append(k["sugg"][0])
            suggested_activities.append(k["sugg"][0])
            suggested_modalities.append(k["sugg"][1])
        else:
            suggested_activities.append("-")
            suggested_modalities.append("-")
        if len(k["sugg"]) > 0 and k["sugg"][0] in last_step:
            equal_to_last_interaction = equal_to_last_interaction + 1
        if s == sim_param["nr_steps"] - 1:
            last_step = curr_step
            curr_step = []
            s = 0
        s = s + 1

    return exp_time, model.schedule.agents[1].rules_credits, model.schedule.agents[
        1].creative_controller.getRules(), actions_credits, predicted_happiness, measured_happiness, actual_happiness, forecast_errors, \
           diversity_therapies, covered_therapies_ratio, diversity_activities, covered_activities_ratio, \
           equal_to_last_interaction, suggested_activities, suggested_modalities, contexts, contexts_names
