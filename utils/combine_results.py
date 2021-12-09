"""
Module that can be used to aggregate the results from different experiments csv files
Happiness, below, is used as an alternative term for Feedback
"""

import os
from statistics import mean, stdev

import pandas as pd

""" Set the following variables to combine the appropriate results """
res_name = "200random"  # an arbitrary name to use for the combined files
resfold = os.path.abspath('../results/synthetitcpatients/random/star')  # change to the right results folder
comb_type = "Global"
# comb_type = "Feedback"
# comb_type = "Repetitions"

""" Setting up the configuration of the module based on the chosen.
Below this line you should not need to modify anything
"""
sheet_to_combine = ""
if comb_type == "Global":
    sheet_to_combine = "Global"

if comb_type == "Feedback":
    dict_h = {}
    sheet_to_combine = "Happiness"
    columns_not_to_use_as_id = ["Step", "sunny", "Suggested Activity", "Suggested Modality", "Predicted Happiness",
                                "Actual Happiness", "Forecast Error"]
    column_to_merge = "Actual Happiness"
    first_col_to_avoid = "Step"
    col_names = []

if comb_type == "Repetitions":
    dict_h = {}
    sheet_to_combine = "Happiness"
    columns_not_to_use_as_id = ["Step", "sunny", "Suggested Activity", "Suggested Modality", "Predicted Happiness",
                                "Actual Happiness", "Forecast Error"]
    column_to_merge = "Suggested Activity"
    first_col_to_avoid = "Step"
    number_of_past_sugg = [14]
    res_name = res_name + "-" + str(number_of_past_sugg).replace("[", "").replace("]", "").replace(",", "-")

""" Combining the results"""
files = os.listdir(resfold)
df_total = pd.DataFrame()
filenr = 0
for file in files:  # loop through Excel files
    if file.startswith("res") and file.endswith('.xlsx'):
        print(str(filenr) + ": " + str(file))
        excel_file = pd.ExcelFile(os.path.join(resfold, file))
        sheets = [sheet_to_combine]
        for sheet in sheets:  # loop through sheets inside an Excel file
            df = excel_file.parse(sheet_name=sheet)
            if comb_type == "Global":
                df_total = df_total.append(df)
            if comb_type == "Feedback":
                sim_happ = df[column_to_merge]
                first_col_to_avoid_loc = df.columns.get_loc(first_col_to_avoid)
                ignore = [df.columns.get_loc("trial")]
                indeces = [x for x in range(first_col_to_avoid_loc) if x not in ignore]
                id_exp = str(list(df.iloc[0, indeces]))
                if not id_exp in dict_h:
                    dict_h[id_exp] = []
                    for i in range(len(sim_happ)):
                        dict_h[id_exp].append([sim_happ[i]])
                else:
                    for i in range(len(sim_happ)):
                        dict_h[id_exp][i].append(sim_happ[i])

                col_names = []
                for c in indeces:
                    col_names.append(df.columns[c])
            if comb_type == "Repetitions":
                sim_sugg = df[column_to_merge]
                first_col_to_avoid_loc = df.columns.get_loc(first_col_to_avoid)
                ignore = []
                indeces = [x for x in range(first_col_to_avoid_loc) if x not in ignore]
                id_exp = str(list(df.iloc[0, indeces]))
                dict_h[id_exp] = []
                for n in range(len(number_of_past_sugg)):
                    dict_h[id_exp].append([])
                    dict_h[id_exp][n] = [[]]
                    for i in range(len(sim_sugg)):  # for each row (suggestions)
                        activity = sim_sugg[i]
                        count = 0
                        for j in range(max(0, i - number_of_past_sugg[n]), i):
                            if sim_sugg[j] == activity:
                                count = count + 1
                        dict_h[id_exp][n][0].append(count)
                    dict_h[id_exp][n][0] = [mean(dict_h[id_exp][n][0]), stdev(dict_h[id_exp][n][0]),
                                            sum(dict_h[id_exp][n][0])]
                col_names = []
                for c in indeces:
                    col_names.append(df.columns[c])
        filenr = filenr + 1

""" Preparing to write the aggregate data to file """
if comb_type == "Feedback":
    listoflists = []
    for exp in dict_h:
        exp_l = (exp.replace("[", "").replace("]", "")).split(",")
        for step in range(len(dict_h[exp])):
            listoflists.append(
                exp_l + [step, mean(dict_h[exp][step]), stdev(dict_h[exp][step]), sum(dict_h[exp][step])])
    df_total = pd.DataFrame.from_records(listoflists, columns=col_names + ["Step", "Average Actual Happiness",
                                                                           "Stdev Actual Happiness",
                                                                           "Sum Actual Happiness"])
if comb_type == "Repetitions":
    listoflists = []
    for exp in dict_h:
        exp_l = (exp.replace("[", "").replace("]", "")).split(",")
        listofrep = []
        listofrepnames = []
        for n in range(len(number_of_past_sugg)):
            listofrepnames = listofrepnames + ["Average " + str(number_of_past_sugg[n]) + " Rep",
                                               "Stdev " + str(number_of_past_sugg[n]) + " Rep",
                                               "Sum " + str(number_of_past_sugg[n]) + " Rep"]
            for step in range(len(dict_h[exp][n])):
                listofrep = listofrep + dict_h[exp][n][step]
        listoflists.append(exp_l + listofrep)
    df_total = pd.DataFrame.from_records(listoflists, columns=col_names + listofrepnames)

""" Writing the results on file """
df_total.to_csv(os.path.join(resfold, "combined_file_" + res_name + "_" + str(comb_type) + ".csv"), index=False)
