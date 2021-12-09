""" Module containing utility functions used for the experimentation """

import os

import numpy
import pandas as pd
import string
import random
from sklearn.metrics import mean_squared_error

import sys
from os import system


def getStatsPerInteraction(measures, nr_steps_interaction):
    """ Function that aggregates the collected simulation data per interaction"""
    avg_v = []
    std_v = []
    step_v = []
    s = 0
    for v in measures:
        step_v.append(v)
        if s == nr_steps_interaction - 1:
            avg_v.append(numpy.mean(step_v))
            std_v.append(numpy.std(step_v))
            step_v = []
            s = 0
        else:
            s = s + 1
    if len(step_v) > 0:
        avg_v.append(numpy.mean(step_v))
        std_v.append(numpy.std(step_v))

    avg_v = numpy.array(avg_v)
    std_v = numpy.array(std_v)
    return avg_v, std_v


def transformResultsIntoDataFrames(allNames, exp_param, exp_param_list, exp_time, rules_credits, rules, actions_credits,
                                   predicted_happiness, measured_happiness, actual_happiness, forecast_errors, diversity_therapies,
                                   covered_therapies_ratio, diversity_activities, covered_activities_ratio,
                                   equal_to_last_interaction, suggested_activities, suggested_modalities,
                                   contexts, context_names):
    """ Utility function that transforms the results from the experimentation into a dataframe to be
    saved as an excel file
    """
    exp_res_happ = []
    exp_res_happint = []
    exp_res_glob = []
    exp_res_div = []
    exp_res_rules = []

    avg_ah, std_ah = getStatsPerInteraction(actual_happiness, exp_param["nr_steps"])
    avg_h, std_h = getStatsPerInteraction(measured_happiness, exp_param["nr_steps"])
    avg_ph, std_ph = getStatsPerInteraction(predicted_happiness, exp_param["nr_steps"])
    avg_e, std_e = getStatsPerInteraction(forecast_errors, exp_param["nr_steps"])

    """ Storing the results for later writing to file"""
    # Happiness
    for i in range(len(predicted_happiness)):
        # happiness_results.append(
        #     exp_param_list + [i, predicted_happiness[i], measured_happiness[i], forecast_errors[i]])
        exp_res_happ.append(
            exp_param_list + [i] + contexts[i] + [suggested_activities[i], suggested_modalities[i],
                                                  predicted_happiness[i],
                                                  actual_happiness[i],
                                                  measured_happiness[i], forecast_errors[i]])
    # Happiness per Interaction
    for i in range(len(avg_ph)):
        # happiness_interaction_results.append(
        #     exp_param_list + [i, avg_ph[i], std_ph[i], avg_h[i], std_h[i], avg_e[i], std_e[i]])
        exp_res_happint.append(
            exp_param_list + [i, avg_ph[i], std_ph[i], avg_ah[i], std_ah[i], avg_h[i], std_h[i], avg_e[i], std_e[i]])

    # Diversity
    for i in range(len(diversity_therapies)):
        # diversity_results.append(
        #     exp_param_list + [i, diversity_therapies[i], covered_therapies_ratio[i], diversity_activities[i],
        #                      covered_activities_ratio[i]])
        exp_res_div.append(
            exp_param_list + [i, diversity_therapies[i], covered_therapies_ratio[i], diversity_activities[i],
                              covered_activities_ratio[i]])
    # Rules
    for r in rules_credits:
        # rules_results.append(
        #     exp_param_list + [str(r), r in [str(x) for x in rules], rules_credits[r]])
        exp_res_rules.append(
            exp_param_list + [str(r), r in [str(x) for x in rules], rules_credits[r]])
    # Global results
    # global_results.append(exp_param_list + [exp_time, numpy.mean(measured_happiness),
    #                                         mean_squared_error(measured_happiness, predicted_happiness),
    #                                         equal_to_last_interaction])
    exp_res_glob.append(exp_param_list + [exp_time, numpy.mean(actual_happiness), numpy.mean(measured_happiness),
                                          mean_squared_error(measured_happiness, predicted_happiness),
                                          equal_to_last_interaction])

    # print(exp_res_happ)
    df_happiness = pd.DataFrame(exp_res_happ,
                                columns=allNames + ['Step'] + context_names + ['Suggested Activity',
                                                                               'Suggested Modality',
                                                                               'Predicted Happiness',
                                                                               'Actual Happiness',
                                                                               'Measured Happiness',
                                                                               'Forecast Error (Pred-Meas)'])
    df_happiness_interaction = pd.DataFrame(exp_res_happint,
                                            columns=allNames + ['Interaction',
                                                                'Interaction Avg Predicted Happiness',
                                                                'Interaction STD Predicted Happiness',
                                                                'Interaction Avg Actual Happiness',
                                                                'Interaction STD Actual Happiness',
                                                                'Interaction Avg Measured Happiness',
                                                                'Interaction STD Measured Happiness',
                                                                'Interaction Avg Forecast Error',
                                                                'Interaction STD Forecast Error'])
    df_diversity = pd.DataFrame(exp_res_div,
                                columns=allNames + ['Step',
                                                    'Diversity Therapies',
                                                    'Covered Therapies Ratio',
                                                    'Diversity Activities',
                                                    'Covered Activities Ratio'])
    df_rules = pd.DataFrame(exp_res_rules,
                            columns=allNames + ['Rule', 'IsFinal', 'Credit'])
    df_global = pd.DataFrame(exp_res_glob,
                             columns=allNames + ['Exp Time', 'Overall Avg Actual Happiness', 'Overall Avg Measured Happiness', 'MSE',
                                                 'Equal to Last Interaction'])
    return [df_happiness, df_happiness_interaction, df_diversity, df_rules, df_global], ["Happiness",
                                                                                         "Happiness by Interaction",
                                                                                         "Diversity", "Rules", "Global"]


def writeToFile(results_folder, filename, list_of_dataframes, list_of_sheets_names):
    """ Function that writes a datagrame into an excel file """
    if not len(list_of_dataframes) == len(list_of_sheets_names):
        print(
            "Warning!! During writing to files, the list of dataframes does not matches with the list of sheet names!")
        print("Skipping these results...")
    else:
        with pd.ExcelWriter(results_folder + filename + '.xlsx') as writer:
            for i in range(len(list_of_dataframes)):
                list_of_dataframes[i].to_excel(writer, sheet_name=list_of_sheets_names[i], index=False)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """ Function that returns an string unique id """
    return ''.join(random.choice(chars) for _ in range(size))


