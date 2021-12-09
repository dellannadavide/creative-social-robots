""" Module containing utility functions used by the different Agents """

import re
import numpy
from random import Random


def importDataFromFiles(ta_details, variables_details, param, preferences_details=None):
    """ Function that imports data from input excel files.
    Data include:
    - fuzzy variables (used as contextual/input var, actiivity var, feedback var) and their possible values
    - (optional) rules to use to initialize the rule bases of the controllers of the agents (from preference file)
    """

    """ Variables """
    contextual_var = list(variables_details.loc[variables_details['vartype'] == "context", 'varname'].unique())
    person_var = list(variables_details.loc[variables_details['vartype'] == "person", 'varname'].unique())
    eval_var = list(variables_details.loc[variables_details['vartype'] == "evaluation", 'varname'].unique())

    variables_crisp_inputs = {}
    var_possible_val = {}

    for c in contextual_var+person_var:
        min = float(variables_details.loc[(variables_details['varname'] == c), 'c_lval_1'].iloc[0])
        max = float(variables_details.loc[(variables_details['varname'] == c), 'c_lval_3'].iloc[0])
        # step = (max-min)/(5.0-1)
        var_possible_val[c] = [
            min,
            # min+step,
            (max - min)/2.0,
            # ((max - min)/2.0)+step,
            max,
        ]
        variables_crisp_inputs[c] = var_possible_val[c][2]  # default val is the middle val
    for c in eval_var:
        min = float(variables_details.loc[(variables_details['varname'] == c), 'c_lval_1'].iloc[0])
        max = float(variables_details.loc[(variables_details['varname'] == c), 'c_lval_3'].iloc[0])
        step = (max-min)/(5.0-1)
        var_possible_val[c] = [
            min,
            min+step,
            (max - min)/2.0,
            ((max - min)/2.0)+step,
            max,
        ]
        variables_crisp_inputs[c] = var_possible_val[c][2]  # default val is the middle val

    ta = list(ta_details['ta_id'].unique())
    ta = [str(i) for i in ta]
    weights_therapeutic_interventions = {}
    for t in ta:
        weights_therapeutic_interventions[t] = 1 #todo fixed for now

    action_var = list(ta_details['sar_action'].unique())
    action_var = [str(i) for i in action_var]
    actions_to_ti = {}
    ta_to_actions = {}
    for a in action_var:
        variables_crisp_inputs[a] = 0  # this is just to create the variable, then the value will be replaced
        actions_to_ti[a] = str((ta_details.loc[(ta_details['sar_action'] == a), 'ta_id']).iloc[0])
        if actions_to_ti[a] in ta_to_actions:
            ta_to_actions[actions_to_ti[a]].append(a)
        else:
            ta_to_actions[actions_to_ti[a]] = [a]

    variables_universe = {}
    dimensions_values = {}
    fuzzysets_values = {}

    range_step = {}
    for v in variables_crisp_inputs:
        fuzzysets_values[v] = {}
        if v in action_var:
            action_lval1 = str((ta_details.loc[(ta_details['sar_action'] == v), 'lval_1']).iloc[0])
            action_lval2 = str((ta_details.loc[(ta_details['sar_action'] == v), 'lval_2']).iloc[0])
            action_lval3 = str((ta_details.loc[(ta_details['sar_action'] == v), 'lval_3']).iloc[0])
            action_c_lval1 = float(ta_details.loc[(ta_details['sar_action'] == v), 'c_lval_1'].iloc[0])
            action_c_lval2 = float(ta_details.loc[(ta_details['sar_action'] == v), 'c_lval_2'].iloc[0])
            action_c_lval3 = float(ta_details.loc[(ta_details['sar_action'] == v), 'c_lval_3'].iloc[0])

            range_step[v] = (action_c_lval3 - action_c_lval1) / param["universes_granularity"]

            variables_crisp_inputs[v] = action_c_lval1 - range_step[v]

            variables_universe[v] = numpy.arange(action_c_lval1 - range_step[v], action_c_lval3 + 1,
                                                      range_step[v])

            dimensions_values[v] = [action_lval1, action_lval2, action_lval3]

            fuzzysets_values[v][action_lval1] = [action_c_lval1, action_c_lval1, action_c_lval1,
                                                      action_c_lval2]
            fuzzysets_values[v][action_lval2] = [action_c_lval1, action_c_lval2, action_c_lval2,
                                                      action_c_lval3]
            fuzzysets_values[v][action_lval3] = [action_c_lval2, action_c_lval3, action_c_lval3,
                                                      action_c_lval3]
        else:
            var_minval = var_possible_val[v][0]
            var_middleval = var_possible_val[v][1]
            var_maxval = var_possible_val[v][2]

            if v in eval_var:
                var_minval = var_possible_val[v][0]
                var_secondval = var_possible_val[v][1]
                var_middleval = var_possible_val[v][2]
                var_secondtolastval = var_possible_val[v][3]
                var_maxval = var_possible_val[v][4]

            variables_universe[v] = numpy.arange(var_minval, var_maxval + 1,
                                                      (var_maxval - var_minval) / param["universes_granularity"])

            if v in eval_var:
                dimensions_values[v] = [
                    "very " + str(variables_details.loc[(variables_details['varname'] == v), 'lval_1'].iloc[0]),
                    variables_details.loc[(variables_details['varname'] == v), 'lval_1'].iloc[0],
                    variables_details.loc[(variables_details['varname'] == v), 'lval_2'].iloc[0],
                    variables_details.loc[(variables_details['varname'] == v), 'lval_3'].iloc[0],
                    "very " + variables_details.loc[(variables_details['varname'] == v), 'lval_3'].iloc[0]]
                fuzzysets_values[v][dimensions_values[v][0]] = [var_minval, var_minval, var_minval,
                                                                var_secondval]
                fuzzysets_values[v][dimensions_values[v][1]] = [var_minval, var_secondval, var_secondval,
                                                                var_middleval]
                fuzzysets_values[v][dimensions_values[v][2]] = [var_secondval, var_middleval, var_middleval,
                                                                var_secondtolastval]
                fuzzysets_values[v][dimensions_values[v][3]] = [var_middleval, var_secondtolastval, var_secondtolastval,
                                                                var_maxval]
                fuzzysets_values[v][dimensions_values[v][4]] = [var_secondtolastval, var_maxval, var_maxval,
                                                                var_maxval]
            else:
                dimensions_values[v] = [
                    variables_details.loc[(variables_details['varname'] == v), 'lval_1'].iloc[0],
                    variables_details.loc[(variables_details['varname'] == v), 'lval_2'].iloc[0],
                    variables_details.loc[(variables_details['varname'] == v), 'lval_3'].iloc[0]]
                fuzzysets_values[v][dimensions_values[v][0]] = [var_minval, var_minval, var_minval,
                                                                var_middleval]
                fuzzysets_values[v][dimensions_values[v][1]] = [var_minval, var_middleval, var_middleval,
                                                                var_maxval]
                fuzzysets_values[v][dimensions_values[v][2]] = [var_middleval, var_maxval, var_maxval,
                                                                var_maxval]

    """ Details of possible rules to be created at the beginning of the simulation to initialize controllers """
    rules_details = []
    if not (preferences_details is None):
        """ Section 1. General Preferences"""
        context_var_dictionary = {  # val in the excel: [var for the controller, val for the controller]
            "morning": ["time", "morning"],
            "afternoon": ["time", "afternoon"],
            "evening": ["time", "evening"],
            "sunny": ["weather", "sunny"],
            "cloudy": ["weather", "cloudy"],
            "rainy": ["weather", "rainy"],
            "a free day": ["dayType", "free"],
            "a working day": ["dayType", "working"],
            "a busy day": ["dayType", "busy"]
        }
        eval_var_dictionary = {
            "don't like": ["emotions", "poor"],
            "hate": ["emotions", "very poor"],
            "like": ["emotions", "good"],
            "love": ["emotions", "very good"]
        }

        indeces_activities = [3, 4, 5, 6, 7, 8]
        nr_rows_per_context = 4
        nr_contexts = 9
        first_context_index = 5
        last_pref_index = 40
        indeces_contexts = []
        for i in range(nr_contexts):
            indeces_contexts.append(first_context_index + (i * nr_rows_per_context))

        curr_context = None
        for c in range(first_context_index, last_pref_index + 1):
            cont = str(preferences_details.loc[c, 1])
            if cont == "nan":
                cont = curr_context
            else:
                curr_context = cont
            for a in indeces_activities:
                if not (str(preferences_details.loc[c, a]) == "nan"):
                    c_var = context_var_dictionary[cont][0]
                    c_val = context_var_dictionary[cont][1]
                    try:
                        activity_id = ta_details.loc[
                            (ta_details['description'] == str(preferences_details.loc[c, a])), "sar_action"].item()
                        activity_val = ta_details.loc[
                            (ta_details['description'] == str(preferences_details.loc[c, a])), "lval_2"].item()

                        rule_eval = eval_var_dictionary[str(preferences_details.loc[c, 2])]
                        rule_details = [[[c_var, c_val]], [[str(activity_id), str(activity_val)]], [rule_eval]]
                        rules_details.append(rule_details)
                        # controller_rule = controller.createAndAddRule([[c_var, c_val]], [[str(activity_id), str(activity_val)]], controller_inputs_mf, controller_outputs_mf)
                    except:
                        # print("skipped because could not find the activity in the activity file")
                        pass
        """ Section 2. Specific Preferences """
        index_row_first_specific_pref = 46
        index_col_first_specific_pref = 2
        max_nr_contexts_specific_pref = 3
        nr_rows_per_specific_pref = 13
        nr_col_per_specific_pref = 2
        nr_rows = 5
        nr_cols = 4
        for i in range(nr_rows):
            for j in range(nr_cols):
                file_row = index_row_first_specific_pref + (i * (nr_rows_per_specific_pref+1)) #+1 because there is title in the excel for each preference
                file_col = index_col_first_specific_pref+(j*nr_col_per_specific_pref)
                if not (str(preferences_details.loc[file_row, file_col]) == "nan"): #if the preference is specified
                    rule_contexts = []
                    rule_evals = []
                    for k in range(nr_rows_per_specific_pref): #reading the entire specific preference row by row
                        cell_content = str(preferences_details.loc[file_row + k, file_col])
                        # print(cell_content)
                        if not (cell_content == "nan"):  # first test should be the same as above
                            if k < max_nr_contexts_specific_pref: #if it's a context
                                c_var = context_var_dictionary[cell_content][0]
                                c_val = context_var_dictionary[cell_content][1]
                                rule_contexts.append([c_var, c_val])
                            elif k==max_nr_contexts_specific_pref: #here it's the "type" good/bad
                                rule_eval = eval_var_dictionary[cell_content]
                                rule_evals.append(rule_eval)
                            else: #here's the activities
                                try:
                                    activity_id = ta_details.loc[
                                        (ta_details['description'] == cell_content), "sar_action"].item()
                                    activity_val = ta_details.loc[
                                        (ta_details['description'] == cell_content), "lval_2"].item()
                                    rule_activities = [[str(activity_id), str(activity_val)]]
                                    if (len(rule_contexts) == 0) or (
                                            len(rule_evals) == 0):
                                        pass
                                    else:
                                        rule_details = [rule_contexts, rule_activities, rule_evals]
                                        # print(rule_details)
                                        rules_details.append(rule_details)
                                except:
                                    pass
    return contextual_var, person_var, eval_var, variables_crisp_inputs, var_possible_val, \
           ta, weights_therapeutic_interventions, action_var, actions_to_ti, ta_to_actions, variables_universe, \
           dimensions_values, fuzzysets_values, range_step, rules_details


def sample_inputs(sample, seed, curr_interaction, variables_crisp_val, action_var, fuzzysets_values, variables_universe, rule_antecedent_info=None):
    """ Function that samples the universe of discourse of the possible variables, returning a crisp value for each var.
     If var @sample is False, returns the default value for the variables """
    if sample:
        r = Random()
        r.seed(seed + curr_interaction)
        sampled_inputs = {}
        for v in variables_crisp_val:
            if v in action_var:  # if it's an action I give the 'default value', i.e., there's no sampling to do or actions, because they have to come from the crules, not randomly
                sampled_inputs[v] = variables_crisp_val[v]
            else:
                if v in rule_antecedent_info:  # if instead the variable is part of the input of the rule given by the chromosome, then I sample but instead from all universe form the range of values for the rule values
                    sampled_inputs[v] = r.uniform(min(fuzzysets_values[v][rule_antecedent_info[v]]),
                                                  max(fuzzysets_values[v][rule_antecedent_info[v]]))
                else:
                    sampled_inputs[v] = r.uniform(min(variables_universe[v]), max(variables_universe[v]))

        return sampled_inputs
    return variables_crisp_val

def getDisabledCounterRule(creative_rule_str):
    """
    Function (still not fully supported now) that from a rule IF prem THEN activity=activity_val creates a "counterrule",
    i.e., a rule IF prem THEN activity=disabled
    """
    disabled_rule_str = None
    f = re.search('a_(.+?)]', creative_rule_str)
    if f:  # found action
        found_action = str(f.group(1)) + "]"
        if not 'disabled' in found_action:
            f2 = re.search('\[(.+?)]', found_action)
            if f2:
                action_val = str(f2.group(1))
                disabled_action = found_action.replace(action_val, 'disabled')
                disabled_rule_str = creative_rule_str.replace(found_action, disabled_action)
    return disabled_rule_str

def getDisabledCounterRuleString(rule):
    """
    Function (still not fully supported now) that returns the string of the counterrule of a rule @rule
    """
    creative_rule_str = str(rule)
    disabled_rule_str = None
    consequent = str(rule.consequent[0])
    if not 'disabled' in consequent:
        action_val = rule.consequent[0].term.label
        print("action val is " + str(action_val))
        disabled_action = consequent.replace(action_val, 'disabled')
        print(creative_rule_str)
        disabled_rule_str = creative_rule_str.replace(consequent, disabled_action)
        print(disabled_rule_str)
    return disabled_rule_str

def getSimpleModalityLinguisticInterpretation(ta_details_df, prop_activity, prop_activity_modality):
    """ Function that provides a simple linguistic interpretation of the value/modality of an activity
    by splitting the universe of discourse in 3 parts (assuming the activity has 3 possible values) """
    pref_lval1 = \
        (ta_details_df.loc[(ta_details_df['sar_action'] == prop_activity), 'lval_1']).iloc[0]
    pref_lval2 = \
        (ta_details_df.loc[(ta_details_df['sar_action'] == prop_activity), 'lval_2']).iloc[0]
    pref_lval3 = \
        (ta_details_df.loc[(ta_details_df['sar_action'] == prop_activity), 'lval_3']).iloc[0]
    pref_c_lval1 = \
        ta_details_df.loc[(ta_details_df['sar_action'] == prop_activity), 'c_lval_1'].iloc[0]
    pref_c_lval2 = \
        ta_details_df.loc[(ta_details_df['sar_action'] == prop_activity), 'c_lval_2'].iloc[0]
    pref_c_lval3 = \
        ta_details_df.loc[(ta_details_df['sar_action'] == prop_activity), 'c_lval_3'].iloc[0]
    delta = float((pref_c_lval3 - pref_c_lval1) / 3.0) + 0.01  # todo hardcoded for now
    if prop_activity_modality <= (pref_c_lval1 + delta):
        prop_activity_interpretation = pref_lval1
    elif prop_activity_modality >= (pref_c_lval3 - delta):
        prop_activity_interpretation = pref_lval3
    else:
        prop_activity_interpretation = pref_lval2
    return prop_activity_interpretation

def getSimpleInputLinguisticInterpretation(var_details_df, input):
    """ Function that provides a simple linguistic interpretation of the value of input variables
        by splitting the universe of discourse in 3 parts (assuming the variable has 3 possible values) """
    ling_input = {}
    for i in input:
        pref_lval1 = \
            (var_details_df.loc[(var_details_df['varname'] == i), 'lval_1']).iloc[0]
        pref_lval2 = \
            (var_details_df.loc[(var_details_df['varname'] == i), 'lval_2']).iloc[0]
        pref_lval3 = \
            (var_details_df.loc[(var_details_df['varname'] == i), 'lval_3']).iloc[0]
        pref_c_lval1 = \
            var_details_df.loc[(var_details_df['varname'] == i), 'c_lval_1'].iloc[0]
        pref_c_lval2 = \
            var_details_df.loc[(var_details_df['varname'] == i), 'c_lval_2'].iloc[0]
        pref_c_lval3 = \
            var_details_df.loc[(var_details_df['varname'] == i), 'c_lval_3'].iloc[0]
        delta = float((pref_c_lval3 - pref_c_lval1) / 3.0) + 0.01  # todo hardcoded for now

        if input[i] <= (pref_c_lval1 + delta):
            interpretation = pref_lval1
        elif input[i] >= (pref_c_lval3 - delta):
            interpretation = pref_lval3
        else:
            interpretation = pref_lval2

        ling_input[i] = interpretation
    return ling_input


def getRepetitionCost(action, logInteractions, param):
    """ Function that given an activity @action and the interactions memory returns the repetition cost,
    i.e., the cost of suggesting @action again, given the number of times nr_times that @action was already suggested
    in the last @param[last_n_steps_negative_repetition] steps.
    The cost is computed as   @param[last_n_steps_negative_repetition]^nr_times
    """
    repetition_cost = 1.0
    nr_times = 0.0
    if param['personalize'] and param["consider_repetition_cost"]:
        logint = len(logInteractions)
        minrange = max(0, logint - param["last_n_steps_negative_repetition"])
        for i in range(minrange, logint):
            if action in logInteractions[i]['sugg']:  # the if-in supports a possible empty list
                nr_times = nr_times + 1
        if nr_times > 0 and param["last_n_steps_repetition_halving_cost"] > 0:
            repetition_cost = param[
                                  "last_n_steps_repetition_halving_cost"] ** nr_times  # - repetition_cost # + w_ta# - len(antecedent_info)*self.param["complexity_cost"] #return the mean output
    return repetition_cost